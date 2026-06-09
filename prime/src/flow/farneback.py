"""
PRIME Farneback Optical Flow
Computes dense optical flow between consecutive frames using
the Farneback algorithm (OpenCV, CPU-based, no weights needed).

Replaces RAFT — lighter, faster, zero model overhead.
Output shape matches what FlowResidual and CropBuilder expect: (H, W, 2).
"""

import cv2
import numpy as np
from src.utils.config_loader import Config
from src.utils.logger import get_logger


class FarnebackFlow:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "farneback",
            cfg.get("logging", "log_path", default="logs/prime.log"),
            cfg.get("logging", "level", default="INFO")
        )

        # Resize target — matches expected_flow dimensions from egomotion
        self.frame_width = cfg.get("camera", "resolution", "width", default=1920)
        self.frame_height = cfg.get("camera", "resolution", "height", default=1080)

        # Farneback parameters
        self.pyr_scale = 0.5
        self.levels = 3
        self.winsize = 15
        self.iterations = 3
        self.poly_n = 5
        self.poly_sigma = 1.2

        self.prev_gray = None  # grayscale of previous frame

        self.logger.info(
            f"FarnebackFlow initialised — "
            f"target {self.frame_width}x{self.frame_height}"
        )

    def compute(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Compute dense optical flow between the previous frame and this frame.

        First call stores the frame and returns None (no previous frame yet).
        All subsequent calls return flow (H, W, 2).

        Args:
            frame: BGR frame from camera (any resolution)

        Returns:
            flow: np.ndarray (H, W, 2) float32 — dx and dy per pixel
                  at self.frame_width x self.frame_height resolution.
            None on the first call.
        """
        # Convert to grayscale and resize to target resolution
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.frame_width, self.frame_height))

        if self.prev_gray is None:
            self.prev_gray = gray
            self.logger.debug("First frame stored — flow computation starts next frame")
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0
        )

        self.prev_gray = gray
        return flow  # (H, W, 2)

    def reset(self):
        """Reset previous frame — call at start of each sweep."""
        self.prev_gray = None
        self.logger.info("FarnebackFlow reset — ready for new sweep")

    def flow_magnitude(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel magnitude from flow (H, W, 2).
        Returns (H, W) float32.
        Useful for building the 4th channel of CNN input crops.
        """
        return np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2).astype(np.float32)

    def visualise(self, flow: np.ndarray) -> np.ndarray:
        """
        Convert flow to HSV colour map for debugging.
        Hue = direction, Value = magnitude.
        Returns BGR image (H, W, 3).
        """
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[:, :, 0] = ang * 180 / np.pi / 2
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def __repr__(self):
        return (
            f"FarnebackFlow("
            f"target={self.frame_width}x{self.frame_height}, "
            f"pyr_scale={self.pyr_scale}, levels={self.levels})"
        )
