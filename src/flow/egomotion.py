"""
ARGUS-N Egomotion
Computes expected optical flow from IMU vehicle speed.
This is the ground truth motion — not estimated from pixels.
Subtract this from RAFT flow to get the residual anomaly map.
"""

import numpy as np
from src.utils.config_loader import Config
from src.utils.logger import get_logger


class Egomotion:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "egomotion",
            cfg.get("logging", "log_path", default="logs/argus.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.imu_enabled = cfg.get("imu", "enabled", default=True)
        self.drift_correction = cfg.get("imu", "drift_correction", default=True)

        self.frame_width = cfg.get("raft", "input_size", "width", default=1920)
        self.frame_height = cfg.get("raft", "input_size", "height", default=1080)
        self.fps = cfg.get("camera", "fps", default=60)

        # Camera mount parameters
        # Camera is 30-35cm from ground, facing down
        self.camera_height_m = 0.325       # midpoint of 30-35cm
        self.focal_length_px = 1200.0      # approximate, update per camera spec

        # IMU state
        self.vehicle_speed_ms = 0.0        # metres per second
        self.vehicle_heading_deg = 0.0     # degrees
        self.drift_offset = np.zeros(2)    # accumulated drift correction

        self.logger.info("Egomotion initialised")

    def update_imu(self, speed_kmh: float, heading_deg: float = 0.0):
        """
        Update vehicle state from IMU reading.
        Call this every IMU tick (200Hz).
        speed_kmh: vehicle speed in km/h
        heading_deg: vehicle heading in degrees (0 = straight ahead)
        """
        self.vehicle_speed_ms = speed_kmh / 3.6
        self.vehicle_heading_deg = heading_deg

    def correct_drift(self, gps_anchor_lat: float, gps_anchor_lon: float):
        """
        Reset IMU drift using known GPS anchor point.
        Called at the start of every sweep at runway threshold.
        """
        self.drift_offset = np.zeros(2)
        self.logger.info(
            f"IMU drift corrected at GPS anchor "
            f"({gps_anchor_lat:.6f}, {gps_anchor_lon:.6f})"
        )

    def compute_expected_flow(self) -> np.ndarray:
        """
        Compute the expected optical flow field for the entire frame
        based on current vehicle speed and camera geometry.

        Physics:
        - Vehicle moves forward at speed v (m/s)
        - Camera at height h (m) above ground, facing down
        - At fps F, displacement per frame = v / F metres
        - In pixels: pixel_displacement = (focal_length * displacement) / height

        Returns:
        expected_flow: np.ndarray shape (H, W, 2)
            flow[:,:,0] = dx (horizontal flow per pixel)
            flow[:,:,1] = dy (vertical flow per pixel)
        """
        # Displacement in metres per frame
        displacement_m = self.vehicle_speed_ms / self.fps

        # Displacement in pixels (perspective projection)
        displacement_px = (self.focal_length_px * displacement_m) / self.camera_height_m

        # Expected flow field — uniform translation
        # Vehicle moves forward → runway moves backward in frame (positive dy)
        expected_flow = np.zeros((self.frame_height, self.frame_width, 2), dtype=np.float32)
        expected_flow[:, :, 0] = 0.0               # no horizontal motion (straight ahead)
        expected_flow[:, :, 1] = displacement_px   # vertical flow from forward motion

        # Apply heading correction for slight turns
        if abs(self.vehicle_heading_deg) > 0.5:
            heading_rad = np.radians(self.vehicle_heading_deg)
            expected_flow[:, :, 0] = displacement_px * np.sin(heading_rad)
            expected_flow[:, :, 1] = displacement_px * np.cos(heading_rad)

        return expected_flow

    def get_dynamic_confirmation_window(self) -> int:
        """
        Confirmation window tied to vehicle speed from IMU.
        Faster speed = more frames needed (vehicle covers more ground per frame).
        Slower speed = fewer frames needed.
        Base: 6 frames at 60FPS = 100ms.
        """
        base = self.cfg.get("bytetrack", "confirmation_frames_base", default=6)
        max_speed = self.cfg.get("bytetrack", "max_speed_kmh", default=50)
        min_speed = self.cfg.get("bytetrack", "min_speed_kmh", default=5)

        speed_kmh = self.vehicle_speed_ms * 3.6

        if speed_kmh <= min_speed:
            return max(2, base // 2)
        elif speed_kmh >= max_speed:
            return base
        else:
            # Linear interpolation between min and max speed
            ratio = (speed_kmh - min_speed) / (max_speed - min_speed)
            return max(2, int(base * ratio + (base // 2) * (1 - ratio)))

    def __repr__(self):
        return (
            f"Egomotion("
            f"speed={self.vehicle_speed_ms * 3.6:.1f}km/h, "
            f"heading={self.vehicle_heading_deg:.1f}deg, "
            f"drift_correction={self.drift_correction})"
        )
