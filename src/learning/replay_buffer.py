"""
ARGUS-N Replay Buffer
Stores confirmed FOD frames and uncertain anomaly frames.
Feeds nightly retraining loop.
Stratified — 60% FOD, 40% clean.
Confidence weighted — physical pickup > visual confirm > quick dismiss.
"""

import json
import shutil
import random
from pathlib import Path
from datetime import datetime
from src.utils.config_loader import Config
from src.utils.logger import get_logger


class ReplayBuffer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "replay_buffer",
            cfg.get("logging", "log_path", default="logs/argus.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.buffer_path = Path(cfg.get(
            "learning", "replay_buffer_path",
            default="data/replay_buffer"
        ))
        self.max_size = cfg.get("learning", "replay_buffer_max", default=10000)
        self.fod_ratio = cfg.get("learning", "replay_buffer_ratio", "fod", default=0.6)
        self.clean_ratio = cfg.get("learning", "replay_buffer_ratio", "clean", default=0.4)

        self.confidence_weights = {
            "physical_pickup": cfg.get("learning", "confidence_weights", "physical_pickup", default=1.0),
            "visual_confirm":  cfg.get("learning", "confidence_weights", "visual_confirm", default=0.7),
            "quick_dismiss":   cfg.get("learning", "confidence_weights", "quick_dismiss", default=0.3),
        }

        # Subdirectories
        self.fod_dir = self.buffer_path / "fod"
        self.clean_dir = self.buffer_path / "clean"
        self.pending_dir = self.buffer_path / "pending"
        self.metadata_path = self.buffer_path / "metadata.json"

        self._init_dirs()
        self.metadata = self._load_metadata()

        self.logger.info(
            f"ReplayBuffer ready — "
            f"max={self.max_size} | "
            f"FOD={int(self.fod_ratio*100)}% | "
            f"clean={int(self.clean_ratio*100)}%"
        )

    def _init_dirs(self):
        self.fod_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self) -> dict:
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {"entries": []}

    def _save_metadata(self):
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def add_anomaly_frame(
        self,
        frame,
        detection: dict,
        source: str = "yolo_uncertain"
    ):
        """
        Save uncertain frame to pending directory.
        Awaits operator confirmation before entering training.

        Args:
            frame:     np.ndarray — the patch or full frame
            detection: dict — YOLO detection metadata
            source:    str — where this came from
        """
        import cv2
        filename = f"pending_{self._timestamp()}.jpg"
        filepath = self.pending_dir / filename
        cv2.imwrite(str(filepath), frame)

        entry = {
            "filename": filename,
            "filepath": str(filepath),
            "label": "pending",
            "confidence_weight": 0.0,
            "source": source,
            "detection": detection,
            "timestamp": self._timestamp()
        }

        self.metadata["entries"].append(entry)
        self._save_metadata()
        self.logger.debug(f"Anomaly frame saved to pending: {filename}")

    def confirm_fod(
        self,
        filename: str,
        confirmation_type: str = "visual_confirm"
    ):
        """
        Operator confirms a pending frame as real FOD.
        Moves from pending to fod directory.
        Assigns confidence weight based on confirmation type.

        confirmation_type:
            physical_pickup — operator physically collected the object (1.0)
            visual_confirm  — operator confirmed on screen (0.7)
            quick_dismiss   — operator dismissed without verification (0.3)
        """
        weight = self.confidence_weights.get(confirmation_type, 0.5)

        # Find entry
        for entry in self.metadata["entries"]:
            if entry["filename"] == filename and entry["label"] == "pending":
                src = Path(entry["filepath"])
                dst = self.fod_dir / filename

                if src.exists():
                    shutil.move(str(src), str(dst))

                entry["label"] = "fod"
                entry["filepath"] = str(dst)
                entry["confidence_weight"] = weight
                entry["confirmation_type"] = confirmation_type

                self._save_metadata()
                self._enforce_size_limit()

                self.logger.info(
                    f"FOD confirmed: {filename} | "
                    f"type={confirmation_type} | "
                    f"weight={weight}"
                )
                return

        self.logger.warning(f"Pending frame not found: {filename}")

    def add_clean_frame(self, frame, source: str = "operator_verified"):
        """
        Add a verified clean runway frame to buffer.
        Used to maintain 60/40 FOD/clean ratio.
        """
        import cv2
        filename = f"clean_{self._timestamp()}.jpg"
        filepath = self.clean_dir / filename
        cv2.imwrite(str(filepath), frame)

        entry = {
            "filename": filename,
            "filepath": str(filepath),
            "label": "clean",
            "confidence_weight": 1.0,
            "source": source,
            "timestamp": self._timestamp()
        }

        self.metadata["entries"].append(entry)
        self._save_metadata()
        self.logger.debug(f"Clean frame added: {filename}")

    def _enforce_size_limit(self):
        """
        If buffer exceeds max size, remove oldest low-weight entries first.
        FIFO with confidence weighting — high confidence samples preserved longer.
        """
        confirmed = [e for e in self.metadata["entries"] if e["label"] != "pending"]

        if len(confirmed) <= self.max_size:
            return

        # Sort by confidence weight ascending — remove lowest confidence first
        confirmed.sort(key=lambda x: x["confidence_weight"])
        to_remove = confirmed[:len(confirmed) - self.max_size]

        for entry in to_remove:
            path = Path(entry["filepath"])
            if path.exists():
                path.unlink()
            self.metadata["entries"].remove(entry)

        self._save_metadata()
        self.logger.info(f"Buffer pruned — removed {len(to_remove)} low-confidence entries")

    def sample(self, n: int = 512) -> list:
        """
        Sample n entries from buffer maintaining FOD/clean ratio.
        Weighted sampling — higher confidence entries sampled more often.

        Returns list of entry dicts.
        """
        fod_entries = [e for e in self.metadata["entries"] if e["label"] == "fod"]
        clean_entries = [e for e in self.metadata["entries"] if e["label"] == "clean"]

        n_fod = int(n * self.fod_ratio)
        n_clean = n - n_fod

        # Weighted sampling
        def weighted_sample(entries, k):
            if not entries:
                return []
            k = min(k, len(entries))
            weights = [e.get("confidence_weight", 0.5) for e in entries]
            total = sum(weights)
            if total == 0:
                return random.sample(entries, k)
            probs = [w / total for w in weights]
            indices = np.random.choice(len(entries), size=k, replace=False, p=probs)
            return [entries[i] for i in indices]

        import numpy as np
        sampled_fod = weighted_sample(fod_entries, n_fod)
        sampled_clean = weighted_sample(clean_entries, n_clean)

        sampled = sampled_fod + sampled_clean
        random.shuffle(sampled)

        self.logger.info(
            f"Sampled {len(sampled_fod)} FOD + "
            f"{len(sampled_clean)} clean = "
            f"{len(sampled)} total"
        )

        return sampled

    def stats(self) -> dict:
        """Return buffer statistics."""
        entries = self.metadata["entries"]
        return {
            "total": len(entries),
            "fod": len([e for e in entries if e["label"] == "fod"]),
            "clean": len([e for e in entries if e["label"] == "clean"]),
            "pending": len([e for e in entries if e["label"] == "pending"]),
            "max_size": self.max_size
        }

    def __repr__(self):
        s = self.stats()
        return (
            f"ReplayBuffer("
            f"total={s['total']}, "
            f"fod={s['fod']}, "
            f"clean={s['clean']}, "
            f"pending={s['pending']})"
        )
