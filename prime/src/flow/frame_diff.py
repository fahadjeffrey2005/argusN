"""
PRIME — FrameDiff
Lightweight temporal signal for the CNN 4th channel.

Replaces Farneback optical flow entirely.

Physics basis:
    Stationary FOD produces near-zero frame difference — it doesn't move.
    Moving artefacts (shadows from passing vehicles, glare, strobes, rain)
    produce non-zero difference.  High 4th-channel value = something moved
    = almost certainly not FOD.

This is one subtraction per frame. Essentially free.
Generalises to any runway and any camera because the physics is universal.
"""

import cv2
import numpy as np


class FrameDiff:
    """
    Per-pixel absolute difference between the current frame and the
    previous frame.  Returns a (H, W) float32 magnitude map — directly
    usable as the CNN 4th channel.
    """

    def __init__(self):
        self.prev_gray: np.ndarray | None = None

    def reset(self):
        """Reset state — call when switching to a new video clip."""
        self.prev_gray = None

    def compute(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute absolute frame difference.

        Args:
            frame: BGR frame (H, W, 3) or grayscale (H, W)

        Returns:
            diff: (H, W) float32 in range [0, 255]
                  First call after reset returns a zero map (no prior frame).
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        gray_f = gray.astype(np.float32)

        if self.prev_gray is None:
            self.prev_gray = gray_f
            return np.zeros(gray.shape[:2], dtype=np.float32)

        diff = np.abs(gray_f - self.prev_gray)
        self.prev_gray = gray_f
        return diff
