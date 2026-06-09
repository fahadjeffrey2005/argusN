"""
HAWKEYE Fusion Layer
Confidence-weighted voting across three independent detection components.

Each component casts a vote (0 or 1) per candidate region:
    YOLO fired?          → +1 vote
    Flow flagged?        → +1 vote
    PatchCore > thresh?  → +1 vote

Alert threshold: 2 or more votes required.
This prevents any single noisy component from causing a false alarm.
"""

import numpy as np
import cv2
from src.utils.config_loader import Config
from src.utils.logger import get_logger


def iou(box_a: dict, box_b: dict) -> float:
    """
    Compute IoU between two bounding boxes.
    Both boxes in {x, y, w, h} format.
    """
    ax1, ay1 = box_a["x"], box_a["y"]
    ax2, ay2 = ax1 + box_a["w"], ay1 + box_a["h"]
    bx1, by1 = box_b["x"], box_b["y"]
    bx2, by2 = bx1 + box_b["w"], by1 + box_b["h"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


class HawkeyeFusion:
    """
    Three-component voting fusion for HAWKEYE.

    Workflow per frame:
        1. Receive YOLO detections (full-frame)
        2. Receive flow candidates (from FlowResidual)
        3. Merge into union of candidate regions (IoU-based deduplication)
        4. For each merged candidate:
            a. Cast YOLO vote (did YOLO fire on this region?)
            b. Cast flow vote (did flow flag this region?)
            c. Score with PatchCore, cast anomaly vote
        5. Alert if total votes >= votes_required
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "hawkeye_fusion",
            cfg.get("logging", "log_path", default="logs/hawkeye.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.votes_required = cfg.get("fusion", "votes_required", default=2)
        self.iou_merge_threshold = 0.3   # IoU above this → same object, merge

        self.logger.info(
            f"HawkeyeFusion initialised — "
            f"votes_required={self.votes_required}"
        )

    def _merge_candidates(
        self,
        yolo_candidates: list,
        flow_candidates: list
    ) -> list:
        """
        Union of YOLO and flow candidates.
        Deduplicate overlapping boxes by IoU — if two candidates overlap,
        keep the merged bounding box (union of both).

        Each merged candidate carries which sources contributed to it.

        Returns list of dicts:
            {x, y, w, h, cx, cy, yolo_vote, flow_vote}
        """
        all_candidates = []

        # Tag sources before merging
        for c in yolo_candidates:
            all_candidates.append({**c, "yolo_vote": True, "flow_vote": False})
        for c in flow_candidates:
            # Check if this flow candidate overlaps an existing yolo candidate
            merged = False
            for existing in all_candidates:
                if iou(c, existing) >= self.iou_merge_threshold:
                    # Merge: take union bounding box, mark flow vote
                    x1 = min(existing["x"], c["x"])
                    y1 = min(existing["y"], c["y"])
                    x2 = max(existing["x"] + existing["w"], c["x"] + c["w"])
                    y2 = max(existing["y"] + existing["h"], c["y"] + c["h"])
                    existing["x"] = x1
                    existing["y"] = y1
                    existing["w"] = x2 - x1
                    existing["h"] = y2 - y1
                    existing["cx"] = float(x1 + (x2 - x1) / 2)
                    existing["cy"] = float(y1 + (y2 - y1) / 2)
                    existing["flow_vote"] = True
                    merged = True
                    break
            if not merged:
                all_candidates.append({**c, "yolo_vote": False, "flow_vote": True})

        return all_candidates

    def fuse(
        self,
        frame: np.ndarray,
        yolo_detections: list,
        flow_candidates: list,
        patchcore
    ) -> list:
        """
        Run full fusion across all three components for one frame.

        Args:
            frame:            current BGR frame
            yolo_detections:  list from YOLODetector.detect() — full frame dets
            flow_candidates:  list from FlowResidual._extract_candidates()
            patchcore:        PatchCore instance (already loaded bank)

        Returns:
            alerts: list of confirmed FOD alerts with vote breakdown
                Each alert: {x, y, w, h, cx, cy, votes, yolo_vote, flow_vote,
                              patchcore_vote, patchcore_score}
        """
        # Convert YOLO detections to candidate format
        yolo_candidates = []
        for det in yolo_detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            w = x2 - x1
            h = y2 - y1
            yolo_candidates.append({
                "x": x1, "y": y1, "w": w, "h": h,
                "cx": float(x1 + w / 2), "cy": float(y1 + h / 2),
                "confidence": det.get("confidence", 1.0)
            })

        # Merge all candidates
        merged = self._merge_candidates(yolo_candidates, flow_candidates)

        if not merged:
            return []

        alerts = []
        for candidate in merged:
            yolo_vote = int(candidate["yolo_vote"])
            flow_vote = int(candidate["flow_vote"])

            # PatchCore vote
            patch = patchcore.extract_patch(frame, candidate)
            pc_anomalous, pc_score = patchcore.is_anomalous(patch)
            patchcore_vote = int(pc_anomalous)

            total_votes = yolo_vote + flow_vote + patchcore_vote

            self.logger.debug(
                f"Candidate ({candidate['cx']:.0f},{candidate['cy']:.0f}) — "
                f"YOLO={yolo_vote} FLOW={flow_vote} PC={patchcore_vote} "
                f"(score={pc_score:.3f}) → votes={total_votes}"
            )

            if total_votes >= self.votes_required:
                alerts.append({
                    "x": candidate["x"],
                    "y": candidate["y"],
                    "w": candidate["w"],
                    "h": candidate["h"],
                    "cx": candidate["cx"],
                    "cy": candidate["cy"],
                    "votes": total_votes,
                    "yolo_vote": yolo_vote,
                    "flow_vote": flow_vote,
                    "patchcore_vote": patchcore_vote,
                    "patchcore_score": round(pc_score, 4)
                })

        if alerts:
            self.logger.info(
                f"ALERT — {len(alerts)} FOD confirmed "
                f"({len(merged)} candidates evaluated)"
            )

        return alerts

    def visualise(self, frame: np.ndarray, alerts: list) -> np.ndarray:
        """
        Draw confirmed alerts on frame.
        Box colour indicates vote count: red=3, orange=2.
        Returns annotated copy.
        """
        vis = frame.copy()
        for alert in alerts:
            x1 = alert["x"]
            y1 = alert["y"]
            x2 = x1 + alert["w"]
            y2 = y1 + alert["h"]

            colour = (0, 0, 255) if alert["votes"] == 3 else (0, 140, 255)
            label = (
                f"FOD [{alert['votes']}/3] "
                f"Y={alert['yolo_vote']} "
                f"F={alert['flow_vote']} "
                f"P={alert['patchcore_vote']}"
            )

            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 3)
            cv2.putText(vis, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        return vis

    def __repr__(self):
        return f"HawkeyeFusion(votes_required={self.votes_required})"
