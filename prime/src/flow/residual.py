"""
ARGUS-N Flow Residual
Subtracts expected egomotion flow from RAFT flow.
Anything remaining is stationary on a moving runway — potential FOD.
Applies exclusion mask for known runway lights.
"""

import numpy as np
import cv2
from pathlib import Path
from src.utils.config_loader import Config
from src.utils.logger import get_logger


class FlowResidual:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "residual",
            cfg.get("logging", "log_path", default="logs/argus.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.residual_threshold = cfg.get("flow", "residual_threshold", default=2.5)
        self.min_area = cfg.get("flow", "min_anomaly_area_px", default=10)
        self.max_area = cfg.get("flow", "max_anomaly_area_px", default=50000)

        self.exclusion_enabled = cfg.get("exclusion_mask", "enabled", default=True)
        self.mask_path = cfg.get("exclusion_mask", "mask_path", default="config/runway_mask.png")

        self.exclusion_mask = None
        self._load_exclusion_mask()

        self.logger.info("FlowResidual initialised")

    def _load_exclusion_mask(self):
        """
        Load runway light exclusion mask.
        Mask is a binary image — white = exclude, black = process.
        Generated once at deployment from airport lighting map.
        If not found, no exclusion applied and warning is logged.
        """
        if self.exclusion_enabled and Path(self.mask_path).exists():
            mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
            self.exclusion_mask = (mask > 128).astype(np.uint8)
            self.logger.info(f"Exclusion mask loaded from {self.mask_path}")
        else:
            self.exclusion_mask = None
            if self.exclusion_enabled:
                self.logger.warning(
                    f"Exclusion mask not found at {self.mask_path} "
                    f"— runway lights will not be excluded"
                )

    def compute(
        self,
        raft_flow: np.ndarray,
        expected_flow: np.ndarray
    ) -> tuple:
        """
        Subtract expected flow from RAFT flow.
        Threshold residual magnitude.
        Apply exclusion mask.
        Extract anomaly regions.

        Args:
            raft_flow:     (H, W, 2) actual optical flow from RAFT
            expected_flow: (H, W, 2) predicted flow from IMU egomotion

        Returns:
            residual_map:  (H, W) float32 — magnitude of residual per pixel
            anomaly_mask:  (H, W) uint8  — binary mask of anomaly regions
            candidates:    list of dicts with bounding box per anomaly region
        """

        # Step 1 — Subtract expected from actual
        residual = raft_flow - expected_flow  # (H, W, 2)

        # Step 2 — Compute magnitude of residual per pixel
        residual_magnitude = np.sqrt(
            residual[:, :, 0] ** 2 + residual[:, :, 1] ** 2
        ).astype(np.float32)

        # Step 3 — Threshold — anything above threshold is anomalous
        anomaly_mask = (residual_magnitude > self.residual_threshold).astype(np.uint8)

        # Step 4 — Apply exclusion mask (runway lights, known static regions)
        if self.exclusion_mask is not None:
            exclusion_resized = cv2.resize(
                self.exclusion_mask,
                (anomaly_mask.shape[1], anomaly_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            anomaly_mask = anomaly_mask & (~exclusion_resized & 1)

        # Step 5 — Morphological cleanup
        # Remove noise, fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)

        # Step 6 — Extract connected components as candidate regions
        candidates = self._extract_candidates(anomaly_mask)

        return residual_magnitude, anomaly_mask, candidates

    def _extract_candidates(self, anomaly_mask: np.ndarray) -> list:
        """
        Find connected anomaly regions and return as bounding boxes.
        Filters by min and max area to remove noise and huge false regions.

        Returns list of dicts:
            {x, y, w, h, area, cx, cy}
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            anomaly_mask, connectivity=8
        )

        candidates = []

        # Label 0 is background — skip
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area < self.min_area or area > self.max_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]

            candidates.append({
                "x":  int(x), "y":  int(y),
                "w":  int(w), "h":  int(h),
                "x1": int(x), "y1": int(y),
                "x2": int(x + w), "y2": int(y + h),
                "area": int(area),
                "cx": float(cx), "cy": float(cy)
            })

        if candidates:
            self.logger.debug(f"{len(candidates)} anomaly candidate(s) found")

        return candidates

    def visualise(
        self,
        frame: np.ndarray,
        candidates: list,
        padding: int = 20
    ) -> np.ndarray:
        """
        Draw candidate bounding boxes on frame for debugging.
        Returns annotated frame.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        for c in candidates:
            x1 = max(0, c["x"] - padding)
            y1 = max(0, c["y"] - padding)
            x2 = min(w, c["x"] + c["w"] + padding)
            y2 = min(h, c["y"] + c["h"] + padding)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(
                vis,
                f"anomaly {c['area']}px",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1
            )

        return vis
