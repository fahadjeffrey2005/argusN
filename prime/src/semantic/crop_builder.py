"""
PRIME Crop Builder
Builds 4-channel crops for each candidate region.
Channel 1-3: BGR patch from frame
Channel 4:   flow magnitude map for the same region

Output shape per crop: (4, 128, 128) float32, normalised 0-1.
This is the input format for the MobileNetV3-Small CNN classifier.
"""

import cv2
import numpy as np
from src.utils.config_loader import Config
from src.utils.logger import get_logger


class CropBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "crop_builder",
            cfg.get("logging", "log_path", default="logs/prime.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.crop_size = cfg.get("cnn", "input_size", default=128)
        self.padding = cfg.get("fusion", "patch_padding_px", default=20)

        self.logger.info(
            f"CropBuilder initialised — "
            f"crop_size={self.crop_size}x{self.crop_size}, "
            f"padding={self.padding}px"
        )

    def build(
        self,
        frame: np.ndarray,
        flow_magnitude: np.ndarray,
        candidate: dict
    ) -> np.ndarray | None:
        """
        Build a single 4-channel crop for one candidate.

        Args:
            frame:          BGR frame (H, W, 3)
            flow_magnitude: per-pixel flow magnitude (H, W) float32
            candidate:      dict with x1, y1, x2, y2

        Returns:
            crop_4ch: np.ndarray (4, crop_size, crop_size) float32 [0-1]
            None if the crop region is degenerate (zero area after clamp)
        """
        fh, fw = frame.shape[:2]

        x1 = max(0, candidate["x1"] - self.padding)
        y1 = max(0, candidate["y1"] - self.padding)
        x2 = min(fw, candidate["x2"] + self.padding)
        y2 = min(fh, candidate["y2"] + self.padding)

        if x2 <= x1 or y2 <= y1:
            return None

        # BGR patch
        bgr_patch = frame[y1:y2, x1:x2]
        if bgr_patch.size == 0:
            return None

        # Flow magnitude patch
        flow_patch = flow_magnitude[y1:y2, x1:x2]

        # Resize both to crop_size x crop_size
        bgr_resized = cv2.resize(
            bgr_patch,
            (self.crop_size, self.crop_size),
            interpolation=cv2.INTER_LINEAR
        )
        flow_resized = cv2.resize(
            flow_patch,
            (self.crop_size, self.crop_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalise
        bgr_norm = bgr_resized.astype(np.float32) / 255.0       # (H, W, 3) [0-1]
        flow_norm = flow_resized.astype(np.float32)
        if flow_norm.max() > 0:
            flow_norm = flow_norm / flow_norm.max()              # (H, W) [0-1]

        # Stack into (4, H, W) — channel-first for PyTorch
        # Channels: B, G, R, flow_magnitude
        b = bgr_norm[:, :, 0]
        g = bgr_norm[:, :, 1]
        r = bgr_norm[:, :, 2]

        crop_4ch = np.stack([b, g, r, flow_norm], axis=0).astype(np.float32)
        return crop_4ch

    def build_batch(
        self,
        frame: np.ndarray,
        flow_magnitude: np.ndarray,
        candidates: list
    ) -> list:
        """
        Build 4-channel crops for all candidates.

        Returns:
            crops: list of (crop_4ch, candidate) pairs
                   crop_4ch is None if the region was degenerate
        """
        results = []
        for candidate in candidates:
            crop = self.build(frame, flow_magnitude, candidate)
            results.append((crop, candidate))
        return results

    def save_crop(
        self,
        crop_4ch: np.ndarray,
        path: str
    ):
        """
        Save a 4-channel crop to disk.
        Saves as a 4-channel PNG using channel-last format.
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convert (4, H, W) → (H, W, 4) for imwrite
        crop_hwc = np.transpose(crop_4ch, (1, 2, 0))
        crop_uint8 = (crop_hwc * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(path, crop_uint8)

    def load_crop(self, path: str) -> np.ndarray:
        """
        Load a saved 4-channel crop from disk.
        Returns (4, H, W) float32 [0-1].
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Crop not found: {path}")
        crop_norm = img.astype(np.float32) / 255.0
        return np.transpose(crop_norm, (2, 0, 1))  # (H, W, 4) → (4, H, W)

    def __repr__(self):
        return (
            f"CropBuilder("
            f"crop_size={self.crop_size}, "
            f"padding={self.padding})"
        )
