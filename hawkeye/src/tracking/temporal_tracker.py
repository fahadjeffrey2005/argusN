"""
HAWKEYE Temporal Confirmation Tracker
======================================
The core of HAWKEYE's advantage over YOLOFINETUNE.

Problem with YOLOFINETUNE:
    It fires on every frame independently. A shadow, a runway marking,
    or a texture variation that looks like FOD for 1-2 frames triggers
    a false alert.

HAWKEYE's solution:
    Track every YOLO detection across consecutive frames via IoU matching.
    Only raise an alert when the SAME object has been detected consistently
    for >= confirm_frames consecutive frames.

    Real FODs are stationary objects. The vehicle approaches them and YOLO
    will detect them in many consecutive frames as the camera closes in.

    Transient false flags (shadows moving, reflections, texture hits) appear
    for 1-3 frames and disappear. They never reach the confirmation threshold.

Result:
    Same recall as YOLOFINETUNE on real FODs.
    Dramatically lower false positive rate on transient events.

Config (config.yaml):
    tracker:
      confirm_frames:  4    # consecutive detections needed to confirm FOD
      iou_threshold:   0.35 # IoU to match detection to existing track
      max_miss_frames: 2    # frames a track can miss before being dropped
"""

import numpy as np
from src.utils.config_loader import Config
from src.utils.logger import get_logger


class Track:
    """Single tracked candidate object."""

    _id_counter = 0

    def __init__(self, detection: dict, confirm_frames: int):
        Track._id_counter += 1
        self.track_id       = Track._id_counter
        self.confirm_frames = confirm_frames
        self.hits           = 1
        self.miss           = 0
        self.confirmed      = False
        self.box            = detection   # {x1, y1, x2, y2, confidence, class_name}

    def update(self, detection: dict):
        """Matched detection — refresh box, increment hit counter."""
        self.box  = detection
        self.hits += 1
        self.miss  = 0
        if self.hits >= self.confirm_frames:
            self.confirmed = True

    def mark_missed(self):
        """No matching detection this frame."""
        self.miss += 1

    @property
    def x1(self): return self.box["x1"]
    @property
    def y1(self): return self.box["y1"]
    @property
    def x2(self): return self.box["x2"]
    @property
    def y2(self): return self.box["y2"]
    @property
    def confidence(self): return self.box.get("confidence", 0.0)

    def as_dict(self) -> dict:
        return {
            "track_id":   self.track_id,
            "x1":         self.x1,
            "y1":         self.y1,
            "x2":         self.x2,
            "y2":         self.y2,
            "confidence": self.confidence,
            "hits":       self.hits,
            "confirmed":  self.confirmed,
        }

    def __repr__(self):
        return (f"Track(id={self.track_id}, hits={self.hits}, "
                f"miss={self.miss}, confirmed={self.confirmed})")


class TemporalTracker:
    """
    Tracks YOLO detections across frames.
    Confirms a FOD only after it appears in >= confirm_frames consecutive frames.

    Usage:
        tracker = TemporalTracker(cfg)

        for frame in video:
            detections = yolo.detect(frame)          # list of {x1,y1,x2,y2,conf}
            confirmed  = tracker.update(detections)  # only real FODs

            for fod in confirmed:
                draw_box(frame, fod)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "temporal_tracker",
            cfg.get("logging", "log_path", default="logs/hawkeye.log"),
            cfg.get("logging", "level",    default="INFO"),
        )

        self.confirm_frames = cfg.get("tracker", "confirm_frames",  default=4)
        self.iou_threshold  = cfg.get("tracker", "iou_threshold",   default=0.35)
        self.max_miss       = cfg.get("tracker", "max_miss_frames",  default=2)

        self.tracks = []

        self.logger.info(
            f"TemporalTracker ready — "
            f"confirm={self.confirm_frames} frames, "
            f"iou_thresh={self.iou_threshold}, "
            f"max_miss={self.max_miss}"
        )

    # ── IoU ───────────────────────────────────────────────────────────────

    @staticmethod
    def _iou(track: Track, det: dict) -> float:
        tx1, ty1, tx2, ty2 = track.x1, track.y1, track.x2, track.y2
        dx1, dy1, dx2, dy2 = det["x1"], det["y1"], det["x2"], det["y2"]

        ix1 = max(tx1, dx1)
        iy1 = max(ty1, dy1)
        ix2 = min(tx2, dx2)
        iy2 = min(ty2, dy2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_t = (tx2 - tx1) * (ty2 - ty1)
        area_d = (dx2 - dx1) * (dy2 - dy1)
        union  = area_t + area_d - inter
        return inter / union if union > 0 else 0.0

    # ── Core update ────────────────────────────────────────────────────────

    def update(self, detections: list) -> list:
        """
        Feed current-frame YOLO detections into the tracker.

        Matching:
          - Each detection is matched to the existing track with highest IoU
            (if IoU >= iou_threshold).
          - Unmatched detections → new tracks (hits=1, not yet confirmed).
          - Unmatched tracks → miss count incremented.
          - Tracks with miss > max_miss → dropped.

        Returns:
            confirmed: list of Track.as_dict() for every track that has
                       been confirmed (hits >= confirm_frames).
                       Draw these as FOD alerts.
        """
        matched_det_idxs   = set()
        matched_track_idxs = set()

        # Build IoU matrix
        if self.tracks and detections:
            n_t = len(self.tracks)
            n_d = len(detections)
            iou_mat = np.zeros((n_t, n_d))
            for ti, track in enumerate(self.tracks):
                for di, det in enumerate(detections):
                    iou_mat[ti, di] = self._iou(track, det)

            # Greedy match — highest IoU first
            while True:
                max_val = iou_mat.max()
                if max_val < self.iou_threshold:
                    break
                ti, di = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
                self.tracks[ti].update(detections[di])
                matched_track_idxs.add(ti)
                matched_det_idxs.add(di)
                iou_mat[ti, :] = -1
                iou_mat[:, di] = -1

        # Mark unmatched tracks as missed
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_idxs:
                track.mark_missed()

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det_idxs:
                self.tracks.append(Track(det, self.confirm_frames))
                self.logger.debug(
                    f"New track #{self.tracks[-1].track_id} — "
                    f"box=({det['x1']},{det['y1']},{det['x2']},{det['y2']})"
                )

        # Drop dead tracks
        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        dropped = before - len(self.tracks)
        if dropped:
            self.logger.debug(f"Dropped {dropped} dead track(s)")

        # Return confirmed FODs
        confirmed = [t.as_dict() for t in self.tracks if t.confirmed]

        if confirmed:
            self.logger.info(
                f"FOD CONFIRMED — {len(confirmed)} track(s): "
                f"{[t['track_id'] for t in confirmed]}"
            )

        return confirmed

    def reset(self):
        """Call at the start of each new video sweep."""
        self.tracks = []
        Track._id_counter = 0
        self.logger.info("TemporalTracker reset")

    def __repr__(self):
        return (f"TemporalTracker("
                f"tracks={len(self.tracks)}, "
                f"confirm={self.confirm_frames})")
