"""
ARGUS-N — Adaptive Multi-Pathway Fusion Gate

Combines output from 3 detection pathways:
  A. PatchCore anomaly score + candidate boxes
  B. CLIP FOD probability per region
  C. RAFT flow residual candidates

Gate logic:
  1. FAST PATH  (1-frame confirmation):
     If >= 2 pathways independently flag the same region -> immediate alert.

  2. TEMPORAL PATH (2-3 frame confirmation):
     If only 1 pathway flags a region -> require it to persist across
     2-3 consecutive frames before alerting.

  3. NIR GATE (post-confirmation filter):
     Dynamic threshold = mean + 2sigma of NIR contrast in recent clean regions.
     Filters shadows, cracks, runway markings.

  4. Region overlap:
     Two pathway detections are co-located if IoU >= 0.2.
"""

import numpy as np
import cv2
from typing import List, Optional, Dict, Tuple
from collections import deque


# -- IoU helpers -----------------------------------------------------------
def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(ua, 1e-6)


def _merge_box(a: tuple, b: tuple) -> tuple:
    return (min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3]))


# -- NIR gate (dynamic threshold) ------------------------------------------
class NIRGate:
    """
    Dynamic NIR contrast gate.
    Threshold = mean + 2*sigma of NIR contrast from confirmed-clean regions.
    Adapts automatically to lighting conditions.
    """

    def __init__(self, window: int = 100, initial_threshold: float = 18.0):
        self._window    = window
        self._history   = deque(maxlen=window)
        self._threshold = initial_threshold

    def update_clean(self, contrast_val: float):
        self._history.append(contrast_val)
        if len(self._history) >= 10:
            arr = np.array(self._history)
            self._threshold = float(arr.mean() + 2.0 * arr.std())

    @property
    def threshold(self) -> float:
        return self._threshold

    def compute_contrast(self, frame_nir: np.ndarray, box: tuple) -> float:
        x1, y1, x2, y2 = box
        h, w = frame_nir.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = frame_nir[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        rh, rw = roi.shape[:2]
        cy, cx = rh // 2, rw // 2
        iy1, iy2 = max(0, cy - rh//4), min(rh, cy + rh//4)
        ix1, ix2 = max(0, cx - rw//4), min(rw, cx + rw//4)
        inner = roi[iy1:iy2, ix1:ix2]
        if inner.size == 0 or inner.size == roi.size:
            return float(np.std(roi))
        mask = np.ones((rh, rw), dtype=bool)
        mask[iy1:iy2, ix1:ix2] = False
        border = roi[mask]
        inner_mean  = float(inner.mean())
        border_mean = float(border.mean()) if border.size > 0 else inner_mean
        return abs(inner_mean - border_mean)

    def passes(self, frame_nir: np.ndarray, box: tuple) -> Tuple[bool, float]:
        contrast = self.compute_contrast(frame_nir, box)
        return contrast >= self._threshold, contrast


# -- Temporal buffer --------------------------------------------------------
class _TemporalEntry:
    def __init__(self, box: tuple, pathway: str):
        self.box      = box
        self.pathway  = pathway
        self.count    = 1
        self.last_box = box

    def update(self, box: tuple):
        self.count   += 1
        self.last_box = box
        self.box      = _merge_box(self.box, box)


# -- Main fusion class ------------------------------------------------------
class AdaptiveFusion:
    """
    Multi-pathway fusion gate for ARGUS-N.
    Call fuse() once per frame with outputs from all pathways.
    """

    def __init__(
        self,
        patchcore_threshold: float = 0.5,
        clip_threshold:      float = 0.5,
        iou_min:             float = 0.2,
        temporal_frames:     int   = 3,
        nir_enabled:         bool  = True,
        nir_window:          int   = 100,
        nir_initial_thresh:  float = 18.0,
    ):
        self.patchcore_threshold = patchcore_threshold
        self.clip_threshold      = clip_threshold
        self.iou_min             = iou_min
        self.temporal_frames     = temporal_frames
        self.nir_enabled         = nir_enabled
        self._nir_gate           = NIRGate(window=nir_window,
                                           initial_threshold=nir_initial_thresh)
        self._temporal: List[_TemporalEntry] = []

    def _normalise_patchcore(self, boxes, score) -> List[Dict]:
        if score < self.patchcore_threshold:
            return []
        return [{"box": b, "pathway": "patchcore", "confidence": score} for b in boxes]

    def _normalise_clip(self, clip_results) -> List[Dict]:
        return [
            {"box": r["box"], "pathway": "clip", "confidence": r["fod_prob"]}
            for r in clip_results if r.get("is_fod")
        ]

    def _normalise_flow(self, flow_boxes) -> List[Dict]:
        return [{"box": b, "pathway": "flow", "confidence": 1.0} for b in flow_boxes]

    def _group_by_overlap(self, regions: List[Dict]) -> List[List[Dict]]:
        clusters: List[List[Dict]] = []
        for reg in regions:
            placed = False
            for cluster in clusters:
                if any(_iou(reg["box"], m["box"]) >= self.iou_min for m in cluster):
                    cluster.append(reg)
                    placed = True
                    break
            if not placed:
                clusters.append([reg])
        return clusters

    def _update_temporal(self, box: tuple, pathway: str):
        for entry in self._temporal:
            if entry.pathway == pathway and _iou(entry.last_box, box) >= self.iou_min:
                entry.update(box)
                return entry.count >= self.temporal_frames, entry
        e = _TemporalEntry(box, pathway)
        self._temporal.append(e)
        return False, e

    def _evict_stale_temporal(self, max_age: int = 5):
        self._temporal = [e for e in self._temporal if e.count < max_age]

    def fuse(
        self,
        frame_bgr: np.ndarray,
        frame_nir: Optional[np.ndarray],
        patchcore_boxes:  List[tuple],
        patchcore_score:  float,
        clip_results:     List[Dict],
        flow_boxes:       List[tuple],
        flow_discarded:   bool = False,
    ) -> Tuple[List[Dict], Dict]:
        """
        Run one frame through the full fusion pipeline.

        Returns:
            alerts:   list of confirmed FOD dicts
            metadata: diagnostic dict
        """
        pa = self._normalise_patchcore(patchcore_boxes, patchcore_score)
        pb = self._normalise_clip(clip_results)
        pc = [] if flow_discarded else self._normalise_flow(flow_boxes)

        all_regions = pa + pb + pc

        if not all_regions:
            self._evict_stale_temporal()
            return [], {"candidates": 0, "fast_path": 0, "temporal_path": 0,
                        "nir_rejected": 0, "flow_discarded": flow_discarded}

        clusters = self._group_by_overlap(all_regions)
        alerts = []
        fast_count = temp_count = nir_rejected = 0

        for cluster in clusters:
            pathways = list({r["pathway"] for r in cluster})
            merged_box = cluster[0]["box"]
            for r in cluster[1:]:
                merged_box = _merge_box(merged_box, r["box"])
            mean_conf = float(np.mean([r["confidence"] for r in cluster]))

            should_confirm = is_fast = False

            if len(pathways) >= 2:
                should_confirm = True
                is_fast        = True
                fast_count    += 1
            else:
                confirmed, _ = self._update_temporal(merged_box, pathways[0])
                if confirmed:
                    should_confirm = True
                    temp_count    += 1

            if not should_confirm:
                continue

            nir_contrast = 0.0
            if self.nir_enabled and frame_nir is not None:
                passes, nir_contrast = self._nir_gate.passes(frame_nir, merged_box)
                if not passes:
                    nir_rejected += 1
                    self._nir_gate.update_clean(nir_contrast)
                    continue

            alerts.append({
                "box":          merged_box,
                "pathways":     pathways,
                "confidence":   mean_conf,
                "nir_contrast": nir_contrast,
                "fast_path":    is_fast,
            })

        self._evict_stale_temporal()

        return alerts, {
            "candidates":    len(clusters),
            "fast_path":     fast_count,
            "temporal_path": temp_count,
            "nir_rejected":  nir_rejected,
            "flow_discarded": flow_discarded,
            "nir_threshold": self._nir_gate.threshold,
        }
