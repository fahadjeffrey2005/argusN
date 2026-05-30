"""
ARGUS-N — Pathway A: DINOv2 + PatchCore Anomaly Detector

Architecture:
  - DINOv2 ViT-S/14 backbone (frozen) for patch-level feature extraction
  - Coreset-sampled memory bank (max 2000 patches from clean runway frames)
  - KNN search (k=3) to compute anomaly score per patch
  - Anomaly heatmap upsampled to original resolution

Key design decisions:
  - Memory bank capped at 2000 via greedy coreset — 800x faster than naive 1.6M
  - Stride=14 (native DINOv2 patch size) — no overlap needed
  - Runs on MPS (dev) / CUDA (Jetson production)
  - build_memory_bank() called once at startup on clean runway frames
  - score() returns (anomaly_score: float, heatmap: np.ndarray)

Usage:
    detector = PatchCoreDetector(device="mps")
    detector.build_memory_bank(clean_frames)   # list of BGR np.ndarray
    score, heatmap = detector.score(frame_bgr)
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from typing import List, Optional, Tuple


# ── Constants ──────────────────────────────────────────────────────────────
BANK_MAX       = 2000       # max coreset patches in memory bank
DINO_PATCH     = 14         # DINOv2 ViT-S/14 patch size in pixels
FEAT_DIM       = 384        # DINOv2 ViT-S feature dimension
KNN_K          = 3          # neighbours for anomaly score
INPUT_SIZE     = 518        # DINOv2 native input (37×37 patches = 518px)
ANOMALY_THRESH = 0.5        # default score threshold (calibrate post bank-build)


class PatchCoreDetector:
    """
    Coreset PatchCore anomaly detector backed by DINOv2 ViT-S/14.
    Thread-safe after build_memory_bank() completes.
    """

    def __init__(self, device: str = "mps", bank_path: Optional[str] = None):
        self.device = torch.device(device if device != "mps" or
                                   torch.backends.mps.is_available() else "cpu")
        self.bank: Optional[torch.Tensor] = None   # (N, FEAT_DIM)
        self._model = None
        self._bank_path = Path(bank_path) if bank_path else None

        self._load_model()

        if self._bank_path and self._bank_path.exists():
            self.load_bank(str(self._bank_path))

    # ── Model ──────────────────────────────────────────────────────────────
    def _load_model(self):
        """Load DINOv2 ViT-S/14 from torch.hub (cached after first download)."""
        import os
        # Ensure hub cache stays on the drive, not the Mac system drive
        _drive_cache = str(Path(__file__).resolve().parents[3] / "models" / "cache")
        os.makedirs(_drive_cache, exist_ok=True)
        torch.hub.set_dir(_drive_cache)
        os.environ.setdefault("TORCH_HOME", _drive_cache)

        print("[PatchCore] Loading DINOv2 ViT-S/14 ...")
        self._model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=True,
            verbose=False,
        )
        self._model.eval().to(self.device)
        # Freeze all parameters — we only use it as a feature extractor
        for p in self._model.parameters():
            p.requires_grad_(False)
        print(f"[PatchCore] Model on {self.device}")

    # ── Pre-processing ─────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(frame_bgr: np.ndarray) -> torch.Tensor:
        """BGR np.ndarray → normalised (1,3,518,518) tensor (ImageNet stats)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return ((t - mean) / std).unsqueeze(0)

    # ── Feature extraction ─────────────────────────────────────────────────
    @torch.no_grad()
    def _extract_patches(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Extract per-patch features from one frame.
        Returns (H_patches * W_patches, FEAT_DIM) on CPU.
        """
        x = self._preprocess(frame_bgr).to(self.device)
        # get_intermediate_layers returns a list; index 0 = last layer tokens
        feats = self._model.get_intermediate_layers(x, n=1)[0]  # (1, N_tokens, D)
        # Drop [CLS] token (index 0) — keep spatial patch tokens only
        # For ViT-S/14 on 518px: 37×37 = 1369 patch tokens nominal
        # (some builds also append register tokens — we truncate those later)
        patch_feats = feats[0, 1:, :]
        return patch_feats.cpu()

    # ── Coreset sampling ───────────────────────────────────────────────────
    @staticmethod
    def _coreset_sample(features: torch.Tensor, n: int) -> torch.Tensor:
        """
        Greedy coreset sampling: iteratively pick the patch that is
        furthest from the already-selected set.
        O(total_patches × n) — fast because n=2000 and total<<1M.
        """
        total = features.shape[0]
        if total <= n:
            return features

        # Initialise with a random seed
        selected_idx = [torch.randint(total, (1,)).item()]
        selected     = features[selected_idx]          # (1, D)

        # Distance of every patch to the nearest selected centre
        min_dists = torch.cdist(features, selected).squeeze(1)  # (total,)

        for _ in range(n - 1):
            farthest = torch.argmax(min_dists).item()
            selected_idx.append(farthest)
            new_dists = torch.cdist(
                features, features[farthest:farthest+1]
            ).squeeze(1)
            min_dists = torch.minimum(min_dists, new_dists)

        return features[selected_idx]

    # ── Memory bank ────────────────────────────────────────────────────────
    def build_memory_bank(self, clean_frames: List[np.ndarray]):
        """
        Build a coreset memory bank from a list of clean runway BGR frames.
        Call this once at startup; blocks until complete.

        Args:
            clean_frames: list of BGR np.ndarray (any resolution)
        """
        if not clean_frames:
            raise ValueError("[PatchCore] No frames provided for memory bank.")

        print(f"[PatchCore] Extracting features from {len(clean_frames)} clean frames...")
        all_feats = []
        for i, frame in enumerate(clean_frames):
            feats = self._extract_patches(frame)   # (1369, 384)
            all_feats.append(feats)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(clean_frames)}]")

        all_feats = torch.cat(all_feats, dim=0)    # (N*1369, 384)
        print(f"[PatchCore] Total patches: {all_feats.shape[0]} → coreset to {BANK_MAX}")

        self.bank = self._coreset_sample(all_feats, BANK_MAX)
        print(f"[PatchCore] Memory bank: {self.bank.shape}")

        if self._bank_path:
            self.save_bank(str(self._bank_path))

    def save_bank(self, path: str):
        torch.save(self.bank, path)
        print(f"[PatchCore] Bank saved → {path}")

    def load_bank(self, path: str):
        self.bank = torch.load(path, map_location="cpu")
        print(f"[PatchCore] Bank loaded: {self.bank.shape} from {path}")

    # ── Scoring ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def score(
        self,
        frame_bgr: np.ndarray,
        return_heatmap: bool = True,
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Score a single frame.

        Returns:
            anomaly_score: float in [0, ∞).  Typical clean: <0.3, FOD: >0.5
            heatmap: np.ndarray (H, W) float32, same size as input frame,
                     values in [0, 1].  None if return_heatmap=False.
        """
        if self.bank is None:
            raise RuntimeError("[PatchCore] Memory bank not built. Call build_memory_bank() first.")

        patch_feats = self._extract_patches(frame_bgr)   # (1369, 384)

        # KNN distances to memory bank
        # cdist: (1369, 2000) — done on CPU to avoid VRAM pressure
        dists = torch.cdist(patch_feats, self.bank)      # (1369, 2000)
        knn_dists, _ = torch.topk(dists, KNN_K, dim=1, largest=False)
        patch_scores = knn_dists.mean(dim=1)             # (1369,)

        # Image-level score = max patch score (worst-case anomaly)
        anomaly_score = float(patch_scores.max())

        if not return_heatmap:
            return anomaly_score, None

        # Reshape to spatial grid and upsample to original frame size
        # Compute actual grid from tensor size — handles non-square grids
        # and DINOv2 versions that drop/add tokens.
        total = patch_scores.shape[0]
        n_h = int(total ** 0.5)
        n_w = total // n_h
        patch_scores = patch_scores[:n_h * n_w]
        score_map = patch_scores.reshape(n_h, n_w).numpy()

        h, w = frame_bgr.shape[:2]
        heatmap = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalise to [0, 1] using a soft clip at 2× threshold
        heatmap = np.clip(heatmap / (ANOMALY_THRESH * 2.0), 0.0, 1.0).astype(np.float32)

        return anomaly_score, heatmap

    def get_candidate_regions(
        self,
        frame_bgr: np.ndarray,
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Return bounding boxes of anomalous patch clusters.

        Args:
            frame_bgr: input frame
            threshold:  anomaly score threshold (default: ANOMALY_THRESH)

        Returns:
            list of (x1, y1, x2, y2) in pixel coords of the original frame
        """
        thr = threshold or ANOMALY_THRESH
        patch_feats = self._extract_patches(frame_bgr)
        dists = torch.cdist(patch_feats, self.bank)
        knn_dists, _ = torch.topk(dists, KNN_K, dim=1, largest=False)
        patch_scores = knn_dists.mean(dim=1)

        # Compute actual grid from tensor size — handles non-square grids
        # and DINOv2 versions that drop/add tokens (36x38=1368, 37x37=1369…)
        total = patch_scores.shape[0]
        n_h = int(total ** 0.5)
        n_w = total // n_h
        patch_scores = patch_scores[:n_h * n_w]   # drop any leftover tokens
        score_map = patch_scores.reshape(n_h, n_w).numpy()

        anomalous = (score_map > thr).astype(np.uint8)

        # If more than 40% of all patches are anomalous, the whole scene is
        # unfamiliar (bad bank, lighting change, video cut) — return nothing
        # rather than one giant full-frame box.
        anomaly_ratio = anomalous.sum() / max(anomalous.size, 1)
        if anomaly_ratio > 0.40:
            return []

        h, w = frame_bgr.shape[:2]
        ph = h / n_h
        pw = w / n_w

        boxes = []
        num_labels, labels = cv2.connectedComponents(anomalous)
        for lbl in range(1, num_labels):
            ys, xs = np.where(labels == lbl)
            if len(xs) < 2:
                continue

            # Weighted centroid — pull the box centre toward the hottest patches
            weights = score_map[ys, xs]
            cx_patch = float(np.average(xs, weights=weights))
            cy_patch = float(np.average(ys, weights=weights))

            # Convert centroid to pixel coords
            cx_px = cx_patch * pw + pw / 2
            cy_px = cy_patch * ph + ph / 2

            # Box half-size: proportional to cluster footprint but tightly capped.
            # cluster_r = radius of the cluster in patch units
            cluster_r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0 + 1
            # Convert to pixels, add one-patch padding, cap at 12% of smaller dim
            half_px = min(
                int(cluster_r * max(pw, ph)) + int(DINO_PATCH * 2),
                int(min(h, w) * 0.12)
            )
            half_px = max(half_px, int(DINO_PATCH * 2))  # minimum visible size

            x1 = max(0, int(cx_px) - half_px)
            y1 = max(0, int(cy_px) - half_px)
            x2 = min(w, int(cx_px) + half_px)
            y2 = min(h, int(cy_px) + half_px)

            boxes.append((x1, y1, x2, y2))

        return boxes
