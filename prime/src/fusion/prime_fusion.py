"""
PRIME Fusion Layer
Merges YOLO candidates and flow candidates using IoU matching.
Tags each merged candidate with its source:
  "both"      — YOLO and flow both flagged this region
  "yolo_only" — only YOLO flagged it
  "flow_only" — only flow flagged it

Source tags propagate to the CNN as prior confidence context:
  "both" candidates receive a confidence bonus (configurable).
"""

import numpy as np
from src.utils.config_loader import Config
from src.utils.logger import get_logger


def _iou(a: dict, b: dict) -> float:
    """
    Compute IoU between two candidates.
    Both must have x1, y1, x2, y2.
    """
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = max(1, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    area_b = max(1, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    union_area = area_a + area_b - inter_area

    return inter_area / union_area


def _merge_boxes(a: dict, b: dict) -> dict:
    """
    Merge two candidate boxes into a single bounding box
    by taking the union (min/max of corners).
    """
    x1 = min(a["x1"], b["x1"])
    y1 = min(a["y1"], b["y1"])
    x2 = max(a["x2"], b["x2"])
    y2 = max(a["y2"], b["y2"])
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "x": x1, "y": y1,
        "w": x2 - x1, "h": y2 - y1,
        "cx": (x1 + x2) / 2,
        "cy": (y1 + y2) / 2,
    }


class PrimeFusion:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "prime_fusion",
            cfg.get("logging", "log_path", default="logs/prime.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.iou_threshold = cfg.get("fusion", "iou_match_threshold", default=0.3)

        self.logger.info(
            f"PrimeFusion initialised — iou_threshold={self.iou_threshold}"
        )

    def merge(
        self,
        yolo_candidates: list,
        flow_candidates: list
    ) -> list:
        """
        Match YOLO and flow candidates by IoU.
        Returns merged list with source tags.

        Each output candidate has:
            x1, y1, x2, y2, x, y, w, h, cx, cy
            tag: "both" | "yolo_only" | "flow_only"
            yolo_confidence: float (0 if no YOLO match)

        Args:
            yolo_candidates: list from YOLODetector.detect()
            flow_candidates: list from FlowResidual._extract_candidates()

        Returns:
            merged: list of tagged candidate dicts
        """
        matched_flow = set()
        matched_yolo = set()
        merged = []

        # Match YOLO candidates to flow candidates
        for yi, yc in enumerate(yolo_candidates):
            best_iou = 0.0
            best_fi = -1

            for fi, fc in enumerate(flow_candidates):
                if fi in matched_flow:
                    continue
                iou = _iou(yc, fc)
                if iou > best_iou:
                    best_iou = iou
                    best_fi = fi

            if best_iou >= self.iou_threshold and best_fi >= 0:
                # Both YOLO and flow agree
                fc = flow_candidates[best_fi]
                box = _merge_boxes(yc, fc)
                merged.append({
                    **box,
                    "tag": "both",
                    "yolo_confidence": yc.get("confidence", 0.0),
                    "flow_area": fc.get("area", 0)
                })
                matched_yolo.add(yi)
                matched_flow.add(best_fi)
                self.logger.debug(
                    f"Matched: YOLO[{yi}] ↔ flow[{best_fi}] IoU={best_iou:.2f} → both"
                )
            else:
                # YOLO only
                merged.append({
                    **{k: yc[k] for k in ("x1", "y1", "x2", "y2", "x", "y", "w", "h")},
                    "cx": (yc["x1"] + yc["x2"]) / 2,
                    "cy": (yc["y1"] + yc["y2"]) / 2,
                    "tag": "yolo_only",
                    "yolo_confidence": yc.get("confidence", 0.0),
                    "flow_area": 0
                })
                matched_yolo.add(yi)

        # Remaining flow candidates with no YOLO match
        for fi, fc in enumerate(flow_candidates):
            if fi not in matched_flow:
                merged.append({
                    **{k: fc[k] for k in ("x1", "y1", "x2", "y2", "x", "y", "w", "h")},
                    "cx": fc.get("cx", (fc["x1"] + fc["x2"]) / 2),
                    "cy": fc.get("cy", (fc["y1"] + fc["y2"]) / 2),
                    "tag": "flow_only",
                    "yolo_confidence": 0.0,
                    "flow_area": fc.get("area", 0)
                })

        tag_counts = {
            "both": sum(1 for c in merged if c["tag"] == "both"),
            "yolo_only": sum(1 for c in merged if c["tag"] == "yolo_only"),
            "flow_only": sum(1 for c in merged if c["tag"] == "flow_only"),
        }
        self.logger.debug(
            f"Fusion: {len(merged)} candidates — "
            f"both={tag_counts['both']}, "
            f"yolo_only={tag_counts['yolo_only']}, "
            f"flow_only={tag_counts['flow_only']}"
        )

        return merged

    def __repr__(self):
        return f"PrimeFusion(iou_threshold={self.iou_threshold})"
