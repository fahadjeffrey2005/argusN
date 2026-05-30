"""
ARGUS-N — Bump Detector

Monitors global optical flow standard deviation across each frame.
When the vehicle hits a bump or pothole, the entire flow field spikes —
every pixel appears to move chaotically.  This is distinct from a stationary
FOD (which creates a LOCAL flow anomaly, or zero anomaly if truly stationary).

Strategy:
  - Compute global σ of the flow magnitude each frame
  - Maintain a rolling baseline of the last N frames (normal vehicle motion)
  - If current σ > baseline_mean + BUMP_K × baseline_σ → bump detected
  - When bump detected: discard this frame's flow pathway entirely
    (Pathways A and B still run independently)

Parameters (tunable in config):
  bump_window: int  — rolling baseline window (default 30 frames = 0.5s at 60fps)
  bump_k:      float — sensitivity multiplier (default 3.0)
                       3.0 = only fires on genuine sharp spikes

Usage:
    detector = BumpDetector(window=30, k=3.0)
    is_bump = detector.update(flow_map)  # call once per frame
    if is_bump:
        # skip flow pathway this frame
"""

import numpy as np
from collections import deque
from typing import Optional


class BumpDetector:
    """
    Rolling-window global flow σ monitor.
    Returns True when a bump is detected (flow pathway should be discarded).
    """

    def __init__(self, window: int = 30, k: float = 3.0):
        self.window   = window
        self.k        = k
        self._history: deque = deque(maxlen=window)
        self._last_sigma: float = 0.0
        self._bump_count: int  = 0

    def update(self, flow: Optional[np.ndarray]) -> bool:
        """
        Update the detector with the latest flow map.

        Args:
            flow: (H, W, 2) optical flow array, or None (no flow yet).

        Returns:
            True  → bump detected, discard flow pathway this frame
            False → flow pathway is valid
        """
        if flow is None:
            return False

        # Global flow magnitude per pixel, then σ across the whole frame
        mag = np.linalg.norm(flow, axis=2)        # (H, W)
        sigma = float(np.std(mag))
        self._last_sigma = sigma

        if len(self._history) < 5:
            # Not enough history yet — accumulate silently
            self._history.append(sigma)
            return False

        # Baseline statistics from rolling window
        hist = np.array(self._history)
        baseline_mean = hist.mean()
        baseline_std  = hist.std() + 1e-6          # avoid division by zero

        is_bump = sigma > baseline_mean + self.k * baseline_std

        if is_bump:
            self._bump_count += 1
            # Do NOT add spike to history (would corrupt baseline)
        else:
            self._history.append(sigma)

        return is_bump

    @property
    def last_sigma(self) -> float:
        return self._last_sigma

    @property
    def bump_count(self) -> int:
        return self._bump_count

    def reset(self):
        self._history.clear()
        self._bump_count = 0
        self._last_sigma = 0.0
