"""
ARGUS-N Camera Ingestion
Handles frame capture from USB camera or video file.
Device agnostic — works on Mac MPS and Ubuntu CUDA.
"""

import cv2
import time
from pathlib import Path
from src.utils.config_loader import Config
from src.utils.logger import get_logger


class CameraIngestion:
    def __init__(self, cfg: Config, camera_index: int = 0):
        self.cfg = cfg
        self.camera_index = camera_index
        self.logger = get_logger(
            "ingestion",
            cfg.get("logging", "log_path", default="logs/yolofinetune.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.width = cfg.get("camera", "resolution", "width", default=1920)
        self.height = cfg.get("camera", "resolution", "height", default=1080)
        self.fps = cfg.get("camera", "fps", default=60)
        self.input_mode = cfg.get("camera", "input_mode", default="usb")
        self.video_path = cfg.get("camera", "video_file_path", default=None)

        self.cap = None
        self._connect()

    def _connect(self):
        if self.input_mode == "video_file" and self.video_path:
            if not Path(self.video_path).exists():
                raise FileNotFoundError(f"Video file not found: {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)
            self.logger.info(f"Source: video file → {self.video_path}")
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.logger.info(f"Source: USB camera index {self.camera_index}")

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera or video source")

        self.logger.info(f"Ingestion ready — {self.width}x{self.height} @ {self.fps}fps")

    def warmup(self, warmup_frames: int = 30):
        """Burn warmup_frames before sweep starts."""
        self.logger.info(f"Warmup — burning {warmup_frames} frames")
        for _ in range(warmup_frames):
            self.cap.read()
        self.logger.info("Warmup complete — pipeline ready")

    def read(self):
        """
        Read one frame.
        Returns (success: bool, frame: np.ndarray)
        """
        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning("Frame read failed — end of stream or camera disconnect")
        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.logger.info("Camera released")

    def __iter__(self):
        """Allow use as iterator in pipeline loop."""
        return self

    def __next__(self):
        ret, frame = self.read()
        if not ret:
            self.release()
            raise StopIteration
        return frame

    def __repr__(self):
        return (
            f"CameraIngestion("
            f"mode={self.input_mode}, "
            f"index={self.camera_index}, "
            f"res={self.width}x{self.height}, "
            f"fps={self.fps})"
        )
