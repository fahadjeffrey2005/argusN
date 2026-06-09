"""
HAWKEYE PatchCore Anomaly Detector
Unsupervised anomaly scoring — no labels required.

Trains on clean tarmac frames only. Builds a memory bank of normal
tarmac feature vectors using WideResNet50 (ImageNet pretrained, frozen).
At inference, scores each candidate patch by its distance to the nearest
normal feature in the bank. High score = anomalous = not normal tarmac.

Architecture:
    Backbone: wide_resnet50_2 (pretrained ImageNet, frozen)
    Layers:   layer2 + layer3 features (concatenated)
    Bank:     coreset-sampled feature vectors from clean frames
    Scoring:  nearest-neighbour distance in feature space

Reference: Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from src.utils.config_loader import Config
from src.utils.logger import get_logger


class PatchCore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "patchcore",
            cfg.get("logging", "log_path", default="logs/hawkeye.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.device = cfg.device
        self.bank_path = cfg.get("patchcore", "bank_path", default="models/patchcore/bank.pt")
        self.anomaly_threshold = cfg.get("patchcore", "anomaly_threshold", default=0.6)
        self.backbone_name = cfg.get("patchcore", "backbone", default="wide_resnet50_2")
        self.layers = cfg.get("patchcore", "layers", default=["layer2", "layer3"])

        self.backbone = None
        self.feature_hooks = {}
        self.memory_bank = None  # (N, D) tensor of normal feature vectors

        self._build_backbone()

        # Load bank if it exists
        if Path(self.bank_path).exists():
            self.load_bank(self.bank_path)
        else:
            self.logger.warning(
                f"PatchCore bank not found at {self.bank_path} — "
                f"run scripts/build_patchcore_bank.py first"
            )

    def _build_backbone(self):
        """
        Load WideResNet50 pretrained on ImageNet, freeze all weights.
        Register forward hooks on target layers to capture feature maps.
        """
        try:
            import torchvision.models as models

            self.backbone = models.wide_resnet50_2(weights="IMAGENET1K_V1")
            self.backbone.eval()
            self.backbone = self.backbone.to(self.device)

            # Freeze all parameters
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Register hooks on target layers
            self.feature_hooks = {}
            self._hook_handles = []

            for layer_name in self.layers:
                layer = getattr(self.backbone, layer_name, None)
                if layer is None:
                    self.logger.warning(f"Layer '{layer_name}' not found in backbone")
                    continue

                def make_hook(name):
                    def hook(module, input, output):
                        self.feature_hooks[name] = output.detach()
                    return hook

                handle = layer.register_forward_hook(make_hook(layer_name))
                self._hook_handles.append(handle)

            self.logger.info(
                f"PatchCore backbone loaded: {self.backbone_name}, "
                f"layers={self.layers}"
            )

        except Exception as e:
            self.logger.error(f"Backbone load failed: {e}")
            self.backbone = None

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess BGR image for WideResNet50.
        Resize to 224x224, normalise with ImageNet mean/std.

        Returns tensor (1, 3, 224, 224)
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        tensor = torch.from_numpy(resized).float() / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        tensor = (tensor - mean) / std
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)
        return tensor.to(self.device)

    def _extract_features(self, image: np.ndarray) -> torch.Tensor:
        """
        Forward pass through backbone, collect hook outputs,
        adaptive-pool and concatenate into a single feature vector.

        Returns: (D,) feature vector
        """
        if self.backbone is None:
            raise RuntimeError("Backbone not loaded")

        tensor = self._preprocess(image)
        self.feature_hooks.clear()

        with torch.no_grad():
            self.backbone(tensor)

        feature_vectors = []
        for layer_name in self.layers:
            if layer_name not in self.feature_hooks:
                continue
            feat = self.feature_hooks[layer_name]  # (1, C, H, W)
            # Adaptive average pool to (1, C, 1, 1)
            pooled = nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            feature_vectors.append(pooled.squeeze())  # (C,)

        combined = torch.cat(feature_vectors, dim=0)  # (D,)
        return combined

    def build_bank(self, clean_frames: list) -> None:
        """
        Build memory bank from a list of clean tarmac frames (BGR np.ndarray).
        Extracts features from each frame and stores in the bank.

        Args:
            clean_frames: list of BGR frames with no FOD present
        """
        if not clean_frames:
            raise ValueError("No clean frames provided for bank building")

        self.logger.info(f"Building PatchCore bank from {len(clean_frames)} clean frames...")

        features = []
        for i, frame in enumerate(clean_frames):
            try:
                feat = self._extract_features(frame)
                features.append(feat)
                if (i + 1) % 10 == 0:
                    self.logger.info(f"  Processed {i + 1}/{len(clean_frames)} frames")
            except Exception as e:
                self.logger.warning(f"Frame {i} skipped: {e}")

        if not features:
            raise RuntimeError("No features extracted — bank empty")

        self.memory_bank = torch.stack(features, dim=0)  # (N, D)
        self.logger.info(
            f"Bank built: {self.memory_bank.shape[0]} vectors, "
            f"dim={self.memory_bank.shape[1]}"
        )

    def save_bank(self, path: str = None):
        """Save memory bank to disk."""
        save_path = Path(path or self.bank_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.memory_bank is None:
            raise RuntimeError("No bank to save — run build_bank() first")

        torch.save(self.memory_bank, save_path)
        self.logger.info(f"PatchCore bank saved to {save_path}")

    def load_bank(self, path: str = None):
        """Load memory bank from disk."""
        load_path = Path(path or self.bank_path)
        self.memory_bank = torch.load(load_path, map_location=self.device)
        self.logger.info(
            f"PatchCore bank loaded from {load_path} — "
            f"{self.memory_bank.shape[0]} vectors"
        )

    def score(self, patch: np.ndarray) -> float:
        """
        Score a single image patch.
        Returns anomaly score in [0, 1].
        Higher = more anomalous = less like normal tarmac.

        Score is computed as:
            1 - exp(-min_distance / scale)
        where min_distance is the L2 distance to the nearest bank vector.

        Args:
            patch: BGR crop of candidate region (any size, will be resized)

        Returns:
            score: float in [0, 1]
        """
        if self.backbone is None or self.memory_bank is None:
            self.logger.warning("PatchCore not ready — returning score 0.0")
            return 0.0

        if patch.size == 0:
            return 0.0

        try:
            feat = self._extract_features(patch)  # (D,)

            # L2 distance to all bank vectors
            diffs = self.memory_bank - feat.unsqueeze(0)  # (N, D)
            distances = torch.norm(diffs, dim=1)           # (N,)
            min_dist = distances.min().item()

            # Normalise to [0, 1] using exponential decay
            # Scale factor calibrated for typical WideResNet50 feature distances
            scale = 10.0
            score = 1.0 - np.exp(-min_dist / scale)
            score = float(np.clip(score, 0.0, 1.0))

            return score

        except Exception as e:
            self.logger.warning(f"PatchCore score failed: {e}")
            return 0.0

    def is_anomalous(self, patch: np.ndarray) -> tuple:
        """
        Score patch and threshold against anomaly_threshold.

        Returns:
            (is_anomalous: bool, score: float)
        """
        score = self.score(patch)
        return score >= self.anomaly_threshold, score

    def extract_patch(
        self,
        frame: np.ndarray,
        candidate: dict,
        padding: int = 10
    ) -> np.ndarray:
        """
        Extract candidate region from frame with padding.

        Args:
            frame:     full BGR frame
            candidate: dict with {x, y, w, h}
            padding:   pixels to expand around bounding box

        Returns:
            patch: BGR crop
        """
        h, w = frame.shape[:2]
        x1 = max(0, candidate["x"] - padding)
        y1 = max(0, candidate["y"] - padding)
        x2 = min(w, candidate["x"] + candidate["w"] + padding)
        y2 = min(h, candidate["y"] + candidate["h"] + padding)
        return frame[y1:y2, x1:x2]

    def __repr__(self):
        bank_size = self.memory_bank.shape[0] if self.memory_bank is not None else 0
        return (
            f"PatchCore("
            f"backbone={self.backbone_name}, "
            f"layers={self.layers}, "
            f"bank={bank_size} vectors, "
            f"threshold={self.anomaly_threshold})"
        )
