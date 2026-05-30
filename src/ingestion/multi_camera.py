"""
ARGUS-N Multi-Camera Ingestion
Handles synchronized frame capture from multiple cameras.
PoC: 2x RGB + 1x NIR (or simulated NIR from RGB).
Scales to 8 cameras for full deployment.

Frame sync strategy:
- All cameras read on the same loop tick
- If any camera drops a frame, the last good frame is reused
- NIR can be real camera or simulated from Camera 0 RGB
"""

import cv2
import numpy as np
from pathlib import Path
from src.utils.config_loader import Config
from src.utils.logger import get_logger
from src.ingestion.nir_simulator import NIRSimulator


class CameraStream:
    """Single camera stream wrapper."""

    def __init__(self, index, width: int, height: int, fps: int, label: str = ""):
        self.index = index
        self.label = label
        self.last_frame = None

        if isinstance(index, (str, Path)) and Path(index).exists():
            self.cap = cv2.VideoCapture(str(index))
        else:
            self.cap = cv2.VideoCapture(index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.opened = self.cap.isOpened()

    def read(self) -> np.ndarray:
        """Read one frame. Returns last good frame on failure."""
        if not self.opened:
            return self.last_frame

        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.last_frame = frame
        return self.last_frame

    def release(self):
        if self.cap:
            self.cap.release()

    def __repr__(self):
        return f"CameraStream(index={self.index}, label={self.label}, open={self.opened})"


class MultiCameraIngestion:
    """
    Manages multiple camera streams synchronously.

    Modes:
        usb         — reads from physical USB cameras by index
        video_file  — reads from video files (for testing/development)
        simulated   — single video source, NIR simulated from RGB

    Output per tick:
        frames_rgb: list of BGR frames from RGB cameras
        frame_nir:  grayscale NIR frame (real or simulated)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "multi_camera",
            cfg.get("logging", "log_path", default="logs/argus.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.width   = cfg.get("camera", "resolution", "width",  default=1920)
        self.height  = cfg.get("camera", "resolution", "height", default=1080)
        self.fps     = cfg.get("camera", "fps",                  default=60)
        self.mode    = cfg.get("camera", "input_mode",           default="simulated")
        self.video_path = cfg.get("camera", "video_file_path",   default=None)
        self.count   = cfg.get("camera", "count",                default=3)
        self.simulate_nir = cfg.get("camera", "simulate_nir",    default=True)

        self.streams: list[CameraStream] = []
        self.nir_stream: CameraStream = None
        self.nir_simulator = NIRSimulator()
        self._synthetic = False          # set True when no video source
        self._synthetic_frame_count = 0
        self._synthetic_max = 10000      # effectively infinite for the pipeline

        self._init_streams()

    def _init_streams(self):
        if self.mode == "simulated" or self.mode == "video_file":
            # Development mode — all cameras read from same video file
            # NIR is simulated from Camera 0
            if not self.video_path or not Path(self.video_path).exists():
                self.logger.warning(
                    "No video file set — generating synthetic concrete-texture frames. "
                    "Pass --source <video.mp4> or set camera.video_file_path in config "
                    "to run on real footage."
                )
                self._synthetic = True
                return

            # Primary RGB stream (Camera 0)
            self.streams.append(CameraStream(
                self.video_path,
                self.width, self.height, self.fps,
                label="RGB_0"
            ))
            self.logger.info(f"Stream 0 (RGB): {self.video_path}")

            # Secondary RGB streams — same source, offset reader
            for i in range(1, self.count if not self.simulate_nir else self.count - 1):
                stream = CameraStream(
                    self.video_path,
                    self.width, self.height, self.fps,
                    label=f"RGB_{i}"
                )
                self.streams.append(stream)
                self.logger.info(f"Stream {i} (RGB): {self.video_path}")

            if self.simulate_nir:
                self.logger.info("NIR: simulated from RGB_0")
            else:
                nir_path = cfg.get("camera", "nir_video_path", default=self.video_path)
                self.nir_stream = CameraStream(
                    nir_path,
                    self.width, self.height, self.fps,
                    label="NIR"
                )
                self.logger.info(f"NIR stream: {nir_path}")

        elif self.mode == "usb":
            # Production mode — physical cameras
            # Layout: indices 0..N-2 are RGB, index N-1 is NIR
            for i in range(self.count - 1):
                stream = CameraStream(
                    i, self.width, self.height, self.fps,
                    label=f"RGB_{i}"
                )
                self.streams.append(stream)
                self.logger.info(f"Stream {i} (RGB): USB camera {i}")

            if self.simulate_nir:
                self.logger.info("NIR: simulated from RGB_0")
            else:
                nir_idx = self.count - 1
                self.nir_stream = CameraStream(
                    nir_idx, self.width, self.height, self.fps,
                    label="NIR"
                )
                self.logger.info(f"NIR stream: USB camera {nir_idx}")

        self.logger.info(
            f"MultiCamera ready — "
            f"{len(self.streams)} RGB stream(s), "
            f"NIR={'simulated' if self.simulate_nir else 'hardware'}"
        )



    def _make_synthetic_frame(self) -> np.ndarray:
        """
        Generate a realistic-ish concrete-texture frame (no real camera needed).
        Used when no video source is configured, so the pipeline can still
        build a PatchCore memory bank and test the detection loop.
        """
        h, w = self.height, self.width
        # Base: dark grey concrete colour
        base = np.full((h, w, 3), 90, dtype=np.uint8)
        # Add Gaussian noise to simulate texture
        noise = np.random.normal(0, 18, (h, w, 3)).astype(np.int16)
        frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Subtle horizontal banding (tarmac grain)
        for y in range(0, h, np.random.randint(20, 60)):
            intensity = np.random.randint(-10, 10)
            frame[y:y+2, :] = np.clip(
                frame[y:y+2, :].astype(np.int16) + intensity, 0, 255
            ).astype(np.uint8)
        return frame

    def read(self) -> tuple:
        """
        Read one synchronized frame from all cameras.

        Returns:
            frames_rgb: list of BGR np.ndarray, one per RGB camera
            frame_nir:  grayscale np.ndarray — NIR (real or simulated)
        """
        # Synthetic mode — no real camera or video file
        if self._synthetic:
            self._synthetic_frame_count += 1
            frame = self._make_synthetic_frame()
            frame_nir = self.nir_simulator.simulate(frame)
            return [frame], frame_nir

        frames_rgb = []
        for stream in self.streams:
            frame = stream.read()
            if frame is not None:
                frames_rgb.append(frame)

        if not frames_rgb:
            return [], None

        # NIR — real hardware or simulated from RGB_0
        if self.simulate_nir:
            frame_nir = self.nir_simulator.simulate(frames_rgb[0])
        else:
            raw_nir = self.nir_stream.read() if self.nir_stream else None
            if raw_nir is not None:
                frame_nir = cv2.cvtColor(raw_nir, cv2.COLOR_BGR2GRAY)
            else:
                frame_nir = self.nir_simulator.simulate(frames_rgb[0])
                self.logger.warning("NIR camera read failed — falling back to simulation")

        return frames_rgb, frame_nir

    def warmup(self, warmup_frames: int = 60):
        """Burn warmup frames to flush camera buffer."""
        self.logger.info(f"Warmup — burning {warmup_frames} frames")
        for _ in range(warmup_frames):
            self.read()
        self.logger.info("Warmup complete")

    def release(self):
        for stream in self.streams:
            stream.release()
        if self.nir_stream:
            self.nir_stream.release()
        self.logger.info("All camera streams released")

    def __iter__(self):
        return self

    def __next__(self):
        frames_rgb, frame_nir = self.read()
        if not frames_rgb or frames_rgb[0] is None:
            self.release()
            raise StopIteration
        return frames_rgb, frame_nir

    def set_source(self, path: str):
        """Override video source — re-init streams with new path."""
        self.video_path = path
        self.streams = []
        self.nir_stream = None
        self._synthetic = False
        self.mode = "video_file"
        self._init_streams()

    def __repr__(self):
        return (
            f"MultiCameraIngestion("
            f"mode={self.mode}, "
            f"streams={len(self.streams)}, "
            f"nir={'sim' if self.simulate_nir else 'hw'})"
        )
