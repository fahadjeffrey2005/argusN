"""
enhanced/src/filtering/post_filter.py
Identical to SHARP's post_filter — copied as-is.
"""

from typing import List
from ..detection.detector import Detection


class PostFilter:
    def __init__(self, min_box_area: int = 64,
                 max_box_fraction: float = 0.25,
                 min_aspect: float = 0.15,
                 max_aspect: float = 6.0):
        self.min_box_area      = min_box_area
        self.max_box_fraction  = max_box_fraction
        self.min_aspect        = min_aspect
        self.max_aspect        = max_aspect

    def apply(self, dets: List[Detection], roi_area: int) -> List[Detection]:
        max_allowed_area = int(roi_area * self.max_box_fraction)
        kept = []
        for d in dets:
            if d.area < self.min_box_area:
                continue
            if d.area > max_allowed_area:
                continue
            if d.aspect < self.min_aspect:
                continue
            if d.aspect > self.max_aspect:
                continue
            kept.append(d)
        return kept

    def explain_rejection(self, d: Detection, roi_area: int) -> str:
        max_allowed_area = int(roi_area * self.max_box_fraction)
        if d.area < self.min_box_area:
            return f"too small ({d.area}px²)"
        if d.area > max_allowed_area:
            return f"too large ({d.area}px²)"
        if d.aspect < self.min_aspect:
            return f"aspect too low ({d.aspect:.2f})"
        if d.aspect > self.max_aspect:
            return f"aspect too high ({d.aspect:.2f})"
        return "passed"
