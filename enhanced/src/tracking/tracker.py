"""
enhanced/src/tracking/tracker.py
----------------------------------
ENHANCED temporal tracker — SHARP tracker + camera motion compensation.

THE KEY UPGRADE:
  SHARP's tracker loses small FOD when the camera moves because a 10×10px
  bolt that shifts 8 pixels between frames gets near-zero IoU with its track.
  Even the center-distance fallback can fail with aggressive camera movement.

  ENHANCED adds camera motion compensation (CMC):
  Before matching detections to tracks each frame, we:
    1. Extract corner features from the previous frame (Shi-Tomasi corners)
    2. Track them to the current frame via Lucas-Kanade optical flow (sparse)
    3. Estimate the affine camera transform (rotation + translation + scale)
    4. Warp ALL existing track positions using that transform

  Result: tracks "follow" the camera movement, so even a tiny bolt that moved
  because the camera panned still has high match score with its predicted position.
  This is the same principle Siddhart uses with his GPU-based motion compensator —
  ours runs on CPU with OpenCV and costs <2ms per frame.
"""

import math
from typing import List, Optional, Tuple
import cv2
import numpy as np
from ..detection.detector import Detection


class _Track:
    _next_id: int = 0

    def __init__(self, det: Detection, confirm_frames: int, peak_conf_min: float = 0.0):
        _Track._next_id += 1
        self.tid              = _Track._next_id
        self.det              = det
        self.hits             = 1
        self.miss             = 0
        self.confirmed        = False
        self.just_confirmed   = False   # True only on the frame it FIRST confirms
        self.confirm_frames   = confirm_frames
        self.peak_conf_min    = peak_conf_min
        self.peak_conf        = det.conf

    def update(self, det: Detection):
        self.det            = det
        self.hits          += 1
        self.miss           = 0
        self.peak_conf      = max(self.peak_conf, det.conf)
        self.just_confirmed = False
        if not self.confirmed:
            if self.hits >= self.confirm_frames and self.peak_conf >= self.peak_conf_min:
                self.confirmed      = True
                self.just_confirmed = True   # fires exactly once per unique FOD object

    def mark_missed(self):
        self.miss           = self.miss + 1
        self.just_confirmed = False

    def warp(self, M: np.ndarray):
        """Apply affine matrix M to this track's bounding box center, update box."""
        if M is None:
            return
        cx = (self.det.x1 + self.det.x2) / 2.0
        cy = (self.det.y1 + self.det.y2) / 2.0
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        warped = cv2.transform(pt, M)[0][0]
        dx = warped[0] - cx
        dy = warped[1] - cy
        # Shift box by (dx, dy), keep same size
        self.det = Detection(
            x1=int(self.det.x1 + dx), y1=int(self.det.y1 + dy),
            x2=int(self.det.x2 + dx), y2=int(self.det.y2 + dy),
            conf=self.det.conf, cls=self.det.cls,
        )


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
    diag     = math.sqrt(frame_w ** 2 + frame_h ** 2)
    max_dist = max_dist_frac * diag
    dcx = (d.x1 + d.x2) / 2.0
    dcy = (d.y1 + d.y2) / 2.0
    dist = math.sqrt((t.det.cx - dcx) ** 2 + (t.det.cy - dcy) ** 2)
    if dist >= max_dist:
        return 0.0
    return 1.0 - (dist / max_dist)


def _match_score(t: _Track, d: Detection,
                 frame_w: int, frame_h: int,
                 small_area: int, max_dist_frac: float) -> float:
    iou      = _iou(t, d)
    is_small = (t.det.area < small_area) or (d.area < small_area)
    if not is_small:
        return iou
    cd = _center_dist_score(t, d, frame_w, frame_h, max_dist_frac)
    return max(iou, cd)


class CameraMotionCompensator:
    """
    Estimates frame-to-frame camera motion using sparse optical flow.
    Returns a 2×3 affine matrix M such that:
        new_position = M @ [old_x, old_y, 1]
    Cost: <2ms per frame on CPU.
    """

    def __init__(self, max_corners: int = 200,
                 quality: float = 0.01,
                 min_distance: int = 30):
        self.max_corners  = max_corners
        self.quality      = quality
        self.min_distance = min_distance
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self):
        self._prev_gray = None

    def update(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Feed the current frame. Returns affine matrix M on success, None on first frame
        or if not enough feature matches found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        M = None
        if self._prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(
                self._prev_gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality,
                minDistance=self.min_distance,
            )
            if pts is not None and len(pts) >= 4:
                pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray, pts, None
                )
                good_prev = pts[status.ravel() == 1]
                good_next = pts_next[status.ravel() == 1]
                if len(good_prev) >= 4:
                    M, _ = cv2.estimateAffinePartial2D(good_prev, good_next)

        self._prev_gray = gray
        return M


class EnhancedTracker:
    """
    Temporal tracker with camera motion compensation.

    All parameters from SHARP plus:
    camera_compensation : bool   — enable CMC (default True)
    comp_max_corners    : int    — feature points for optical flow
    comp_quality        : float  — corner quality
    comp_min_distance   : int    — min px between corners
    """

    def __init__(self,
                 confirm_frames: int       = 3,
                 small_confirm_frames: int = 3,
                 small_area_px: int        = 1000,
                 iou_threshold: float      = 0.25,
                 max_miss: int             = 2,
                 max_dist_frac: float      = 0.05,
                 peak_conf_min: float      = 0.35,
                 camera_compensation: bool = True,
                 comp_max_corners: int     = 200,
                 comp_quality: float       = 0.01,
                 comp_min_distance: int    = 30):

        self.confirm_frames       = confirm_frames
        self.small_confirm_frames = small_confirm_frames
        self.small_area_px        = small_area_px
        self.iou_threshold        = iou_threshold
        self.max_miss             = max_miss
        self.max_dist_frac        = max_dist_frac
        self.peak_conf_min        = peak_conf_min
        self._tracks: List[_Track] = []
        self._frame_w: int = 1920
        self._frame_h: int = 1080

        self.use_cmc = camera_compensation
        self.cmc     = CameraMotionCompensator(
            max_corners=comp_max_corners,
            quality=comp_quality,
            min_distance=comp_min_distance,
        ) if camera_compensation else None

    def reset(self, frame_w: int = 1920, frame_h: int = 1080):
        self._tracks    = []
        self._frame_w   = frame_w
        self._frame_h   = frame_h
        _Track._next_id = 0
        if self.cmc:
            self.cmc.reset()

    def _confirm_for(self, det: Detection) -> int:
        if det.area < self.small_area_px:
            return self.small_confirm_frames
        return self.confirm_frames

    def update(self, detections: List[Detection],
               frame: Optional[np.ndarray] = None,
               frame_w: int = None,
               frame_h: int = None) -> List[Detection]:
        """
        Parameters
        ----------
        detections : list of Detection for this frame
        frame      : current BGR frame (required for camera motion compensation)
        """
        fw = frame_w or self._frame_w
        fh = frame_h or self._frame_h

        # Step 1: estimate camera motion and warp all track positions
        if self.use_cmc and self.cmc is not None and frame is not None:
            M = self.cmc.update(frame)
            if M is not None:
                for t in self._tracks:
                    t.warp(M)

        # Step 2: match detections to (now motion-compensated) tracks
        matched_t: set = set()
        matched_d: set = set()

        if self._tracks and detections:
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
                self._tracks.append(_Track(d, self._confirm_for(d), self.peak_conf_min))

        self._tracks = [t for t in self._tracks if t.miss <= self.max_miss]

        confirmed_dets    = [t.det for t in self._tracks if t.confirmed]
        newly_confirmed   = [t for t in self._tracks if t.just_confirmed]

        return confirmed_dets, newly_confirmed
