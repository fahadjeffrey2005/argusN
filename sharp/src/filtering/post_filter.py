"""
sharp/src/filtering/post_filter.py
------------------------------------
Post-tracking filters — NEW in SHARP, not in HAWKEYE.

Applied AFTER the temporal tracker confirms a detection.
Kills remaining false positives that the tracker cannot catch because
they are persistent (e.g. the yellow centerline that broke v2 training).

Filters
-------
1. min_box_area      — tiny boxes are almost always noise, not physical objects
2. max_box_fraction  — a box covering >25% of the ROI is a runway feature, not FOD
3. aspect ratio      — very thin horizontal strips = runway joints / painted lines
"""

from typing import List
from ..detection.detector import Detection


class PostFilter:
    """
    Filters applied to tracker-confirmed detections before they are drawn
    or saved as labels.

    Parameters
    ----------
    min_box_area : int
        Minimum bounding-box area in pixels².  Default 100.
        A 10×10px box at 1080p is clearly sub-object noise.
    max_box_fraction : float
        Maximum fraction of the ROI area a box may cover.
        E.g. 0.25 = if a box is bigger than 25% of the search zone, reject it.
    min_aspect : float
        Minimum width/height ratio.  Very tall/thin = line artifact.
    max_aspect : float
        Maximum width/height ratio.  Very wide/flat = runway stripe or joint.
    """

    def __init__(self, min_box_area: int = 100,
                 max_box_fraction: float = 0.25,
                 min_aspect: float = 0.15,
                 max_aspect: float = 6.0):
        self.min_box_area      = min_box_area
        self.max_box_fraction  = max_box_fraction
        self.min_aspect        = min_aspect
        self.max_aspect        = max_aspect

    def apply(self, dets: List[Detection], roi_area: int) -> List[Detection]:
        """
        Filter a list of confirmed detections.

        Parameters
        ----------
        dets : list of Detection (full-frame coords)
        roi_area : int — pixel area of the ROI (y_end-y_start) * (x_end-x_start)

        Returns filtered list.
        """
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
        """Return a human-readable reason why a detection would be rejected."""
        max_allowed_area = int(roi_area * self.max_box_fraction)
        if d.area < self.min_box_area:
            return f"too small ({d.area}px² < {self.min_box_area})"
        if d.area > max_allowed_area:
            return f"too large ({d.area}px² > {max_allowed_area}, {self.max_box_fraction*100:.0f}% of ROI)"
        if d.aspect < self.min_aspect:
            return f"aspect too low ({d.aspect:.2f} < {self.min_aspect}) — vertical strip"
        if d.aspect > self.max_aspect:
            return f"aspect too high ({d.aspect:.2f} > {self.max_aspect}) — horizontal stripe"
        return "passed"
