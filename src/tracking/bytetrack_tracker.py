"""
ARGUS-N ByteTrack Tracker
Assigns unique ID to every detected object.
Uses Kalman filter to predict next position.
Dynamic confirmation window tied to vehicle speed.
Object must persist N frames to be confirmed as real FOD.
"""

import numpy as np
from collections import defaultdict
from src.utils.config_loader import Config
from src.utils.logger import get_logger


class Track:
    """Single tracked object."""

    _id_counter = 0

    def __init__(self, detection: dict, confirmation_window: int):
        Track._id_counter += 1
        self.track_id = Track._id_counter
        self.confirmation_window = confirmation_window
        self.hit_count = 1
        self.miss_count = 0
        self.confirmed = False
        self.active = True

        # Kalman filter state
        # State: [cx, cy, w, h, vx, vy]
        self.state = np.array([
            (detection["x1"] + detection["x2"]) / 2,  # cx
            (detection["y1"] + detection["y2"]) / 2,  # cy
            detection["x2"] - detection["x1"],         # w
            detection["y2"] - detection["y1"],         # h
            0.0,                                        # vx
            0.0                                         # vy
        ], dtype=np.float32)

        self.last_detection = detection

    def predict(self):
        """
        Predict next position using constant velocity model.
        cx += vx, cy += vy
        """
        self.state[0] += self.state[4]  # cx += vx
        self.state[1] += self.state[5]  # cy += vy

    def update(self, detection: dict):
        """Update track with new matched detection."""
        cx = (detection["x1"] + detection["x2"]) / 2
        cy = (detection["y1"] + detection["y2"]) / 2
        w  = detection["x2"] - detection["x1"]
        h  = detection["y2"] - detection["y1"]

        # Update velocity
        self.state[4] = cx - self.state[0]
        self.state[5] = cy - self.state[1]

        # Update position and size
        self.state[0] = cx
        self.state[1] = cy
        self.state[2] = w
        self.state[3] = h

        self.hit_count += 1
        self.miss_count = 0
        self.last_detection = detection

        # Confirm if persisted long enough
        if self.hit_count >= self.confirmation_window:
            self.confirmed = True

    def mark_missed(self):
        """Called when no detection matched this track in current frame."""
        self.miss_count += 1

    def predicted_box(self) -> dict:
        """Return predicted bounding box from current state."""
        cx, cy, w, h = self.state[:4]
        return {
            "x1": int(cx - w / 2),
            "y1": int(cy - h / 2),
            "x2": int(cx + w / 2),
            "y2": int(cy + h / 2)
        }

    def __repr__(self):
        return (
            f"Track(id={self.track_id}, "
            f"hits={self.hit_count}, "
            f"misses={self.miss_count}, "
            f"confirmed={self.confirmed})"
        )


class ByteTrackTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "bytetrack",
            cfg.get("logging", "log_path", default="logs/argus.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.track_thresh = cfg.get("bytetrack", "track_thresh", default=0.5)
        self.match_thresh = cfg.get("bytetrack", "match_thresh", default=0.8)
        self.track_buffer = cfg.get("bytetrack", "track_buffer", default=30)
        self.confirmation_window = cfg.get("bytetrack", "confirmation_frames_base", default=6)
        self.dynamic_window = cfg.get("bytetrack", "dynamic_window", default=True)

        self.tracks = []

        self.logger.info(
            f"ByteTrack ready — "
            f"confirmation window: {self.confirmation_window} frames"
        )

    def _iou(self, box1: dict, box2: dict) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1["x1"], box2["x1"])
        y1 = max(box1["y1"], box2["y1"])
        x2 = min(box1["x2"], box2["x2"])
        y2 = min(box1["y2"], box2["y2"])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
        area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _match_detections(
        self,
        detections: list
    ) -> tuple:
        """
        Match detections to existing tracks using IoU.
        Returns matched pairs, unmatched detections, unmatched tracks.
        """
        if not self.tracks or not detections:
            return [], detections, self.tracks

        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        # Build IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t_idx, track in enumerate(self.tracks):
            pred_box = track.predicted_box()
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._iou(pred_box, det)

        # Greedy matching — highest IoU first
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self.match_thresh:
                break
            t_idx, d_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched.append((t_idx, d_idx))
            iou_matrix[t_idx, :] = -1
            iou_matrix[:, d_idx] = -1
            if t_idx in unmatched_tracks:
                unmatched_tracks.remove(t_idx)
            if d_idx in unmatched_dets:
                unmatched_dets.remove(d_idx)

        unmatched_det_objs = [detections[i] for i in unmatched_dets]
        unmatched_track_objs = [self.tracks[i] for i in unmatched_tracks]

        return matched, unmatched_det_objs, unmatched_track_objs

    def update(
        self,
        detections: list,
        confirmation_window: int = None
    ) -> tuple:
        """
        Update tracker with new detections from YOLO.
        Returns confirmed FOD tracks and all active tracks.

        Args:
            detections:          list of detection dicts from YOLODetector
            confirmation_window: dynamic window from Egomotion (overrides base)

        Returns:
            confirmed_fods:  list of tracks confirmed as real FOD
            active_tracks:   all currently active tracks
        """
        # Use dynamic window if provided
        window = confirmation_window if confirmation_window else self.confirmation_window

        # Step 1 — Predict all track positions
        for track in self.tracks:
            track.predict()

        # Step 2 — Match detections to tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)

        # Step 3 — Update matched tracks
        for t_idx, d_idx in matched:
            self.tracks[t_idx].update(detections[d_idx])
            self.tracks[t_idx].confirmation_window = window

        # Step 4 — Mark unmatched tracks as missed
        for track in unmatched_tracks:
            track.mark_missed()

        # Step 5 — Create new tracks for unmatched detections
        for det in unmatched_dets:
            new_track = Track(det, window)
            self.tracks.append(new_track)
            self.logger.debug(f"New track created: ID {new_track.track_id}")

        # Step 6 — Remove dead tracks (missed too long)
        self.tracks = [
            t for t in self.tracks
            if t.miss_count <= self.track_buffer
        ]

        # Step 7 — Return confirmed FODs
        confirmed_fods = [t for t in self.tracks if t.confirmed]

        if confirmed_fods:
            self.logger.info(
                f"{len(confirmed_fods)} confirmed FOD(s) — "
                f"track IDs: {[t.track_id for t in confirmed_fods]}"
            )

        return confirmed_fods, self.tracks

    def reset(self):
        """Reset all tracks — call at start of each sweep."""
        self.tracks = []
        Track._id_counter = 0
        self.logger.info("ByteTrack reset — ready for new sweep")

    def visualise(
        self,
        frame: np.ndarray,
        active_tracks: list
    ):
        """
        Draw all active tracks on frame.
        Confirmed tracks in red, unconfirmed in yellow.
        """
        import cv2
        vis = frame.copy()

        for track in active_tracks:
            box = track.predicted_box()
            color = (0, 0, 255) if track.confirmed else (0, 255, 255)
            label = f"ID:{track.track_id} hits:{track.hit_count}"

            cv2.rectangle(
                vis,
                (box["x1"], box["y1"]),
                (box["x2"], box["y2"]),
                color, 2
            )
            cv2.putText(
                vis, label,
                (box["x1"], box["y1"] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1
            )

        return vis

    def __repr__(self):
        return (
            f"ByteTrackTracker("
            f"tracks={len(self.tracks)}, "
            f"confirmation_window={self.confirmation_window}, "
            f"dynamic={self.dynamic_window})"
        )
