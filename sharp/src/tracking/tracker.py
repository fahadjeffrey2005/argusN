"""
sharp/src/tracking/tracker.py
-------------------------------
Temporal tracker — SHARP upgrade over HAWKEYE's TemporalTracker.

KEY IMPROVEMENTS FOR SMALL FOD:
  1. Hybrid matching — IoU primary, center-point distance fallback for small boxes.
     IoU collapses to near-zero for tiny objects with any camera motion.
     A 10×10px bolt shifting 4px between frames: IoU ≈ 0.02 but centers are 4px apart.
     Center-distance matching catches this; IoU alone misses it.

  2. Scale-aware confirm threshold — small objects confirm in fewer frames (default 2)
     than large ones (default 3). A tiny bolt is only detectable at high conf
     for a brief window before it grows as the camera approaches. Requiring 3
     consecutive frames costs you that window.

  3. All parameters exposed and configurable from sharp_config.yaml.
"""

import math
from typing import List
from ..detection.detector import Detection


class _Track:
    _next_id: int = 0

    def __init__(self, det: Detection, confirm_frames: int):
        _Track._next_id += 1
        self.tid            = _Track._next_id
        self.det            = det
        self.hits           = 1
        self.miss           = 0
        self.confirmed      = False
        self.confirm_frames = confirm_frames

    def update(self, det: Detection):
        self.det   = det
        self.hits += 1
        self.miss  = 0
        if self.hits >= self.confirm_frames:
            self.confirmed = True

    def mark_missed(self):
        self.miss += 1

    @property
    def cx(self) -> float:
        return (self.det.x1 + self.det.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.det.y1 + self.det.y2) / 2.0


def _iou(t: _Track, d: Detection) -> float:
    ix1 = max(t.det.x1, d.x1); iy1 = max(t.det.y1, d.y1)
    ix2 = min(t.det.x2, d.x2); iy2 = min(t.det.y2, d.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = t.det.area + d.area - inter
    return inter / (union + 1e-6)


def _center_dist_score(t: _Track, d: Detection,
                       frame_w: int, frame_h: int,
                       max_dist_frac: float = 0.05) -> float:
    """
    Normalised center-point closeness score in [0, 1].
    Returns 1.0 if centers are identical, 0.0 if dist >= max_dist_frac * frame_diagonal.
    max_dist_frac = 0.05 means two boxes can be up to 5% of the frame diagonal apart
    and still be considered the same object.
    """
    diag = math.sqrt(frame_w ** 2 + frame_h ** 2)
    max_dist = max_dist_frac * diag

    dcx = (d.x1 + d.x2) / 2.0
    dcy = (d.y1 + d.y2) / 2.0
    dist = math.sqrt((t.cx - dcx) ** 2 + (t.cy - dcy) ** 2)

    if dist >= max_dist:
        return 0.0
    return 1.0 - (dist / max_dist)


def _match_score(t: _Track, d: Detection,
                 frame_w: int, frame_h: int,
                 small_area: int,
                 max_dist_frac: float) -> float:
    """
    Hybrid score: IoU + center-distance fallback for small objects.
    For objects below small_area threshold, use whichever score is higher.
    For large objects, use IoU only (center-dist is too permissive on big boxes).
    """
    iou = _iou(t, d)
    is_small = (t.det.area < small_area) or (d.area < small_area)
    if not is_small:
        return iou
    # Small object: return best of IoU and center-distance score
    cd = _center_dist_score(t, d, frame_w, frame_h, max_dist_frac)
    return max(iou, cd)


class SharpTracker:
    """
    Temporal confirmation tracker with small-FOD improvements.

    Parameters
    ----------
    confirm_frames : int
        Consecutive frames needed to confirm a large detection. Default 3.
    small_confirm_frames : int
        Consecutive frames needed to confirm a SMALL detection. Default 2.
        Lower because tiny FOD has a short high-confidence window.
    small_area_px : int
        Bounding-box area threshold (px²) below which a detection is "small".
        Default 1000 (~32×32px at 1080p). At 640-inference a bolt at range
        is typically 8×8 to 20×20 px on the cropped region.
    iou_threshold : float
        Minimum match score to link a detection to a track. Default 0.20.
        Slightly lower than HAWKEYE (0.25) to tolerate small-box drift.
    max_miss : int
        Frames a track may be absent before deletion. Default 2.
    max_dist_frac : float
        For center-distance matching, max allowed distance as a fraction of
        the frame diagonal. Default 0.05 (5%). At 1080p that is ~62px.
    """

    def __init__(self,
                 confirm_frames: int       = 3,
                 small_confirm_frames: int = 2,
                 small_area_px: int        = 1000,
                 iou_threshold: float      = 0.20,
                 max_miss: int             = 2,
                 max_dist_frac: float      = 0.05):

        self.confirm_frames       = confirm_frames
        self.small_confirm_frames = small_confirm_frames
        self.small_area_px        = small_area_px
        self.iou_threshold        = iou_threshold
        self.max_miss             = max_miss
        self.max_dist_frac        = max_dist_frac
        self._tracks: List[_Track] = []
        self._frame_w: int = 1920
        self._frame_h: int = 1080

    def reset(self, frame_w: int = 1920, frame_h: int = 1080):
        """Call between videos. Pass frame dimensions for distance normalisation."""
        self._tracks    = []
        self._frame_w   = frame_w
        self._frame_h   = frame_h
        _Track._next_id = 0

    def _confirm_for(self, det: Detection) -> int:
        """Return confirm_frames based on detection size."""
        if det.area < self.small_area_px:
            return self.small_confirm_frames
        return self.confirm_frames

    def update(self, detections: List[Detection],
               frame_w: int = None, frame_h: int = None) -> List[Detection]:
        """
        Feed one frame's detections.
        Returns confirmed detections for this frame.
        """
        fw = frame_w or self._frame_w
        fh = frame_h or self._frame_h

        matched_t: set = set()
        matched_d: set = set()

        if self._tracks and detections:
            # Build score matrix
            scores = [
                [_match_score(t, d, fw, fh, self.small_area_px, self.max_dist_frac)
                 for d in detections]
                for t in self._tracks
            ]

            while True:
                best_score = -1.0
                best_ti = best_di = -1
                for ti in range(len(self._tracks)):
                    for di in range(len(detections)):
                        if ti not in matched_t and di not in matched_d:
                            if scores[ti][di] > best_score:
                                best_score = scores[ti][di]
                                best_ti = ti
                                best_di = di

                if best_score < self.iou_threshold or best_ti < 0:
                    break

                self._tracks[best_ti].update(detections[best_di])
                matched_t.add(best_ti)
                matched_d.add(best_di)
                for k in range(len(detections)):
                    scores[best_ti][k] = -1.0
                for k in range(len(self._tracks)):
                    scores[k][best_di] = -1.0

        for ti, t in enumerate(self._tracks):
            if ti not in matched_t:
                t.mark_missed()

        for di, d in enumerate(detections):
            if di not in matched_d:
                cf = self._confirm_for(d)
                self._tracks.append(_Track(d, cf))

        self._tracks = [t for t in self._tracks if t.miss <= self.max_miss]
        return [t.det for t in self._tracks if t.confirmed]

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    @property
    def confirmed_tracks(self) -> int:
        return sum(1 for t in self._tracks if t.confirmed)
