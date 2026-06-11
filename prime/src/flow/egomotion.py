"""
PRIME Egomotion
Computes expected optical flow from IMU vehicle speed.
This is the ground truth motion — not estimated from pixels.
Subtract this from Farneback flow to get the residual anomaly map.
Copied from hawkeye — identical component shared across both models.
"""

import numpy as np
from src.utils.config_loader import Config
from src.utils.logger import get_logger


class Egomotion:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "egomotion",
            cfg.get("logging", "log_path", default="logs/prime.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.imu_enabled = cfg.get("imu", "enabled", default=False)
        self.fps = cfg.get("camera", "fps", default=60)

        # Camera mount geometry
        self.camera_height_m = cfg.get("egomotion", "camera_height_m", default=0.325)
        self.focal_length_px = cfg.get("egomotion", "focal_length_px", default=1200.0)

        # IMU state — start at simulated speed
        simulated_kmh = cfg.get("imu", "simulated_speed_kmh", default=30.0)
        self.vehicle_speed_ms = simulated_kmh / 3.6
        self.vehicle_heading_deg = 0.0

        self.logger.info(
            f"Egomotion initialised — "
            f"imu={self.imu_enabled}, "
            f"speed={self.vehicle_speed_ms * 3.6:.1f}km/h"
        )

    def update_imu(self, speed_kmh: float, heading_deg: float = 0.0):
        """
        Update vehicle state from IMU reading or manual override.
        speed_kmh: vehicle speed in km/h
        heading_deg: vehicle heading in degrees (0 = straight ahead)
        """
        self.vehicle_speed_ms = speed_kmh / 3.6
        self.vehicle_heading_deg = heading_deg

    def compute_expected_flow(
        self,
        frame_height: int = None,
        frame_width: int = None
    ) -> np.ndarray:
        """
        Compute the expected optical flow field for the entire frame
        based on current vehicle speed and camera geometry.

        Args:
            frame_height: actual frame height in pixels (after ROI crop).
            frame_width:  actual frame width in pixels.

        IMPORTANT: always pass the actual cropped frame dimensions.
        After ROI crop the frame is shorter than the original — passing the
        wrong dimensions causes a shape mismatch with the Farneback flow output.

        Physics:
        - Vehicle moves forward at speed v (m/s)
        - Camera at height h (m) above ground, facing down
        - At fps F, displacement per frame = v / F metres
        - In pixels: pixel_displacement = (focal_length * displacement) / height

        Returns:
            expected_flow: np.ndarray shape (H, W, 2)
                flow[:,:,0] = dx (horizontal)
                flow[:,:,1] = dy (vertical — positive = forward vehicle motion)
        """
        h = frame_height if frame_height is not None else 1080
        w = frame_width  if frame_width  is not None else 1920

        displacement_m  = self.vehicle_speed_ms / self.fps
        displacement_px = (self.focal_length_px * displacement_m) / self.camera_height_m

        expected_flow = np.zeros((h, w, 2), dtype=np.float32)
        expected_flow[:, :, 0] = 0.0               # no horizontal (straight ahead)
        expected_flow[:, :, 1] = displacement_px   # vertical from forward motion

        if abs(self.vehicle_heading_deg) > 0.5:
            heading_rad = np.radians(self.vehicle_heading_deg)
            expected_flow[:, :, 0] = displacement_px * np.sin(heading_rad)
            expected_flow[:, :, 1] = displacement_px * np.cos(heading_rad)

        return expected_flow

    def __repr__(self):
        return (
            f"Egomotion("
            f"speed={self.vehicle_speed_ms * 3.6:.1f}km/h, "
            f"heading={self.vehicle_heading_deg:.1f}deg)"
        )
