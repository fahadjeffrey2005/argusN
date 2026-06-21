"""
sharp/src/detection/detector.py
--------------------------------
YOLO detector wrapper with full ROI crop support.

Improvements over HAWKEYE:
  - imgsz configurable (640 or 1280)
  - left/right crop in addition to top/bottom
  - returns raw pixel coords in full-frame space immediately
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1)

    def as_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "conf": self.conf, "cls": self.cls}


class SharpDetector:
    """
    Wraps a YOLO model with configurable ROI crop and image size.
    All detections are returned in full-frame pixel coordinates.
    """

    def __init__(self, model_path: str, conf: float, imgsz: int, device: str,
                 top_crop: float, bot_crop: float,
                 left_crop: float = 0.0, right_crop: float = 0.0):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.top_crop = top_crop
        self.bot_crop = bot_crop
        self.left_crop = left_crop
        self.right_crop = right_crop
        self.class_names: dict = self.model.names

    def roi_bounds(self, h: int, w: int):
        """Return (y_start, y_end, x_start, x_end) pixel bounds for this frame size."""
        y_start = int(h * self.top_crop)
        y_end   = int(h * (1.0 - self.bot_crop))
        x_start = int(w * self.left_crop)
        x_end   = int(w * (1.0 - self.right_crop))
        return y_start, y_end, x_start, x_end

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO on the ROI-cropped frame.
        Returns detections in full-frame pixel coordinates.
        """
        h, w = frame.shape[:2]
        y_start, y_end, x_start, x_end = self.roi_bounds(h, w)

        cropped = frame[y_start:y_end, x_start:x_end]
        results  = self.model.predict(cropped, conf=self.conf,
                                       imgsz=self.imgsz, verbose=False,
                                       device=self.device)
        dets = []
        for r in results:
            for box in r.boxes:
                cx1, cy1, cx2, cy2 = [int(v) for v in box.xyxy[0].tolist()]
                # Convert cropped → full-frame
                dets.append(Detection(
                    x1=cx1 + x_start, y1=cy1 + y_start,
                    x2=cx2 + x_start, y2=cy2 + y_start,
                    conf=float(box.conf[0]),
                    cls=int(box.cls[0]),
                ))
        return dets
