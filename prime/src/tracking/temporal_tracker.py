"""
PRIME Temporal Confirmation Tracker
=====================================
Copied from HAWKEYE — identical component, updated log path.

PRIME's pipeline:
    YOLO detects candidates → CNN classifies → only FOD-class passes here
    TemporalTracker confirms FODs that persist >= confirm_frames consecutive frames

Because CNN pre-filters semantic false positives (shadows, markings, strobes),
the tracker only sees CNN-confirmed FOD candidates.
Transient misclassifications that slip past the CNN are caught here.

Result: two independent filters — semantic (CNN) + temporal (tracker) —
give PRIME the lowest false positive rate of the three models.
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
        self.box            = detection   # {x1, y1, x2, y2, confidence, ...}

    def update(self, detection: dict):
        self.box   = detection
        self.hits += 1
        self.miss  = 0
        if self.hits >= self.confirm_frames:
            self.confirmed = True

    def mark_missed(self):
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
    Tracks CNN-confirmed FOD candidates across frames.
    Raises alert only after >= confirm_frames consecutive detections.

    Usage:
        tracker = TemporalTracker(cfg)
        for frame in video:
            yolo_dets  = yolo.detect(frame)
            fod_dets   = [d for d in cnn_results if is_fod]
            confirmed  = tracker.update(fod_dets)
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.logger = get_logger(
            "temporal_tracker",
            cfg.get("logging", "log_path",   default="logs/prime.log"),
            cfg.get("logging", "level",      default="INFO"),
        )

        self.confirm_frames = cfg.get("tracker", "confirm_frames",  default=3)
        self.iou_threshold  = cfg.get("tracker", "iou_threshold",   default=0.25)
        self.max_miss       = cfg.get("tracker", "max_miss_frames",  default=2)

        self.tracks = []

        self.logger.info(
            f"TemporalTracker ready — "
            f"confirm={self.confirm_frames} frames, "
            f"iou_thresh={self.iou_threshold}, "
            f"max_miss={self.max_miss}"
        )

    @staticmethod
    def _iou(track: Track, det: dict) -> float:
        tx1, ty1, tx2, ty2 = track.x1, track.y1, track.x2, track.y2
        dx1, dy1, dx2, dy2 = det["x1"], det["y1"], det["x2"], det["y2"]
        ix1 = max(tx1, dx1); iy1 = max(ty1, dy1)
        ix2 = min(tx2, dx2); iy2 = min(ty2, dy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_t = max(1, (tx2 - tx1) * (ty2 - ty1))
        area_d = max(1, (dx2 - dx1) * (dy2 - dy1))
        union  = area_t + area_d - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list) -> list:
        """
        Feed current-frame detections (already CNN-filtered) into the tracker.
        Returns list of confirmed FOD dicts (hits >= confirm_frames).
        """
        matched_det   = set()
        matched_track = set()

        if self.tracks and detections:
            n_t = len(self.tracks)
            n_d = len(detections)
            iou_mat = np.zeros((n_t, n_d))
            for ti, track in enumerate(self.tracks):
                for di, det in enumerate(detections):
                    iou_mat[ti, di] = self._iou(track, det)

            while True:
                max_val = iou_mat.max()
                if max_val < self.iou_threshold:
                    break
                ti, di = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
                self.tracks[ti].update(detections[di])
                matched_track.add(ti)
                matched_det.add(di)
                iou_mat[ti, :] = -1
                iou_mat[:, di] = -1

        for ti, track in enumerate(self.tracks):
            if ti not in matched_track:
                track.mark_missed()

        for di, det in enumerate(detections):
            if di not in matched_det:
                self.tracks.append(Track(det, self.confirm_frames))

        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        dropped = before - len(self.tracks)
        if dropped:
            self.logger.debug(f"Dropped {dropped} stale track(s)")

        confirmed = [t.as_dict() for t in self.tracks if t.confirmed]
        if confirmed:
            self.logger.info(
                f"FOD CONFIRMED — {len(confirmed)} track(s): "
                f"{[t['track_id'] for t in confirmed]}"
            )

        return confirmed

    def reset(self):
        self.tracks = []
        Track._id_counter = 0
        self.logger.info("TemporalTracker reset")

    def __repr__(self):
        return (f"TemporalTracker("
                f"tracks={len(self.tracks)}, "
                f"confirm={self.confirm_frames})")
