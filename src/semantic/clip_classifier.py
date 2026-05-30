"""
ARGUS-N — Pathway B: CLIP Semantic Classifier

Architecture:
  - CLIP ViT-B/32 (OpenAI) — frozen, inference only
  - Receives 96×96 pixel crops of candidate regions (NOT full frames)
  - Multi-prompt ensemble: 8 FOD prompts vs 4 clean prompts
  - Returns FOD probability per crop via softmax over text similarities

Design rationale:
  - Running CLIP on full 1920×1080 frames is pointless — it would average
    over the entire scene.  We crop to regions flagged by Pathway A / flow.
  - 96×96 is chosen to preserve enough texture while fitting CLIP's
    224×224 input via bilinear upsample (CLIP handles this fine).
  - Multi-prompt ensemble reduces sensitivity to prompt wording.
  - Zero-shot: no fine-tuning needed.  CLIP generalises across FOD types,
    lighting conditions, runway colours.

Usage:
    clf = CLIPClassifier(device="mps")
    crops = [frame[y1:y2, x1:x2] for x1,y1,x2,y2 in boxes]
    probs = clf.classify_crops(crops)   # list of float in [0,1]
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import List, Optional

try:
    import clip
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

CROP_SIZE    = 96      # crops are resized to this before feeding CLIP
CLIP_INPUT   = 224     # CLIP's expected input size

# ── Prompt bank ────────────────────────────────────────────────────────────
FOD_PROMPTS = [
    "a photo of foreign object debris on a runway",
    "a piece of metal, rubber, or debris on an airport tarmac",
    "a small object on a runway that could damage an aircraft",
    "debris, litter, or loose material on a concrete runway surface",
    "a wire, bolt, or fragment lying on airport pavement",
    "a foreign object on the ground that is a runway hazard",
    "a rock, screw, plastic, or metal piece on an airfield",
    "an unexpected small object on an otherwise clean runway",
]

CLEAN_PROMPTS = [
    "a photo of a clean empty runway with no debris",
    "a clear airport tarmac surface with no foreign objects",
    "normal runway markings and clean concrete pavement",
    "an empty runway with only pavement markings visible",
]


class CLIPClassifier:
    """
    Zero-shot CLIP classifier for FOD detection on image crops.
    Thread-safe after __init__ completes.
    """

    def __init__(self, device: str = "mps", threshold: float = 0.5):
        if not _CLIP_AVAILABLE:
            raise ImportError(
                "clip not installed.  Run: pip install git+https://github.com/openai/CLIP.git"
            )
        self.threshold = threshold
        self.device = torch.device(
            device if device != "mps" or torch.backends.mps.is_available() else "cpu"
        )
        self._load_model()
        self._encode_prompts()

    # ── Model ──────────────────────────────────────────────────────────────
    def _load_model(self):
        print("[CLIP] Loading ViT-B/32 ...")
        self._model, self._preprocess = clip.load("ViT-B/32", device=self.device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        print(f"[CLIP] Model on {self.device}")

    @torch.no_grad()
    def _encode_prompts(self):
        """Pre-encode all text prompts once — stored as normalised vectors."""
        fod_tokens   = clip.tokenize(FOD_PROMPTS).to(self.device)
        clean_tokens = clip.tokenize(CLEAN_PROMPTS).to(self.device)

        fod_feats   = self._model.encode_text(fod_tokens)    # (8, 512)
        clean_feats = self._model.encode_text(clean_tokens)  # (4, 512)

        # Ensemble: mean of all prompts per class, then normalise
        fod_mean   = F.normalize(fod_feats.mean(dim=0, keepdim=True),   dim=-1)  # (1, 512)
        clean_mean = F.normalize(clean_feats.mean(dim=0, keepdim=True), dim=-1)  # (1, 512)

        # Stack: row 0 = FOD, row 1 = clean
        self._text_feats = torch.cat([fod_mean, clean_mean], dim=0)  # (2, 512)
        print(f"[CLIP] Text prompts encoded ({len(FOD_PROMPTS)} FOD, {len(CLEAN_PROMPTS)} clean)")

    # ── Pre-processing ─────────────────────────────────────────────────────
    def _preprocess_crop(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """
        BGR crop → normalised CLIP tensor (1, 3, 224, 224).
        Resizes to CLIP_INPUT with bilinear interpolation.
        """
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (CLIP_INPUT, CLIP_INPUT), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        return ((t - mean) / std).unsqueeze(0)

    # ── Classification ─────────────────────────────────────────────────────
    @torch.no_grad()
    def classify_crops(
        self,
        crops: List[np.ndarray],
    ) -> List[float]:
        """
        Classify a list of BGR crops.

        Returns:
            list of float: FOD probability in [0, 1] per crop.
            0.0 = definitely clean, 1.0 = definitely FOD.
        """
        if not crops:
            return []

        # Batch all crops together for efficiency
        batch = torch.cat([self._preprocess_crop(c) for c in crops], dim=0)
        batch = batch.to(self.device)

        image_feats = self._model.encode_image(batch)                  # (N, 512)
        image_feats = F.normalize(image_feats, dim=-1)

        # Cosine similarity to text classes
        logits = (image_feats @ self._text_feats.T) * 100.0            # (N, 2)
        probs  = torch.softmax(logits, dim=-1)                         # (N, 2)

        # Return FOD probability (index 0)
        return probs[:, 0].cpu().tolist()

    @torch.no_grad()
    def is_fod(self, crop_bgr: np.ndarray) -> tuple:
        """
        Classify a single crop.

        Returns:
            (is_fod: bool, probability: float)
        """
        prob = self.classify_crops([crop_bgr])[0]
        return prob >= self.threshold, prob

    # ── Batch scoring from boxes ───────────────────────────────────────────
    def score_regions(
        self,
        frame_bgr: np.ndarray,
        boxes: List[tuple],
    ) -> List[dict]:
        """
        Extract crops from boxes in the frame and classify each.

        Args:
            frame_bgr: full resolution frame
            boxes:     list of (x1, y1, x2, y2)

        Returns:
            list of dicts with keys: box, fod_prob, is_fod
        """
        if not boxes:
            return []

        crops = []
        valid_boxes = []
        h, w = frame_bgr.shape[:2]

        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            valid_boxes.append((x1, y1, x2, y2))

        probs = self.classify_crops(crops)

        return [
            {"box": box, "fod_prob": prob, "is_fod": prob >= self.threshold}
            for box, prob in zip(valid_boxes, probs)
        ]
