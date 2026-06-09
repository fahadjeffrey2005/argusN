"""
HAWKEYE Farneback Optical Flow
Dense optical flow between consecutive frames using OpenCV Farneback algorithm.
Extracted from argusN raft_flow.py — RAFT dependency removed entirely.
No weights needed — pure CPU-based OpenCV computation.
"""

import cv2
import numpy as np
from src.utils.config_loader import Config
from src.utils.logger import get_logger


class FarnebackFlow:
    """
    Computes dense optical flow between frame T-1 and frame T.

    Parameters (OpenCV calcOpticalFlowFarneback):
        pyr_scale:  0.5  — image pyramid scale between layers
        levels:     3    — number of pyramid layers
        winsize:    15   — averaging window size (larger = smoother, slower)
        iterations: 3    — iterations at each pyramid level
        poly_n:     5    — pixel neighbourhood for polynomial expansion
        poly_sigma: 1.2  — standard deviation of Gaussian for polynomial expansion
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "farneback",
            cfg.get("logging", "log_path", default="logs/hawkeye.log"),
            cfg.get("logging", "level", default="INFO")
        )

        # Farneback parameters
        self.pyr_scale = 0.5
        self.levels = 3
        self.winsize = 15
        self.iterations = 3
        self.poly_n = 5
        self.poly_sigma = 1.2

        # Store previous frame for inter-frame flow
        self.prev_gray = None

        self.logger.info("FarnebackFlow initialised")

    def compute(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Compute dense optical flow between previous frame and current frame.

        First call stores frame and returns None (no previous frame yet).
        Subsequent calls return flow map (H, W, 2) — dx and dy per pixel.

        Args:
            frame: BGR frame (H, W, 3)

        Returns:
            flow: np.ndarray (H, W, 2) or None on first call
                  flow[:,:,0] = horizontal displacement (dx)
                  flow[:,:,1] = vertical displacement (dy)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

    def magnitude_map(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel flow magnitude from a flow field.

        Args:
            flow: (H, W, 2)

        Returns:
            magnitude: (H, W) float32
        """
        return np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2).astype(np.float32)

    def reset(self):
        """Reset previous frame — call at start of each sweep."""
        self.prev_gray = None
        self.logger.info("FarnebackFlow reset — ready for new sweep")

    def visualise(self, flow: np.ndarray) -> np.ndarray:
        """
        Render flow field as HSV colour wheel image for debugging.
        Hue = direction, Value = magnitude.

        Returns:
            BGR image (H, W, 3)
        """
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 1] = 255

        magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        hsv[:, :, 0] = angle * 180 / np.pi / 2
        hsv[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def __repr__(self):
        return (
            f"FarnebackFlow("
            f"pyr_scale={self.pyr_scale}, "
            f"levels={self.levels}, "
            f"winsize={self.winsize})"
        )
