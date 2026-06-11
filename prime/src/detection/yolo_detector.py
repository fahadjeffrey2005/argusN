"""
PRIME YOLO Detector
Runs YOLOv8 fine-tuned on the full ROI-cropped frame.
Produces candidate bounding boxes that feed into PrimeFusion.
Copied from hawkeye — full-frame independent component.
"""

import cv2
import numpy as np
from pathlib import Path
from src.utils.config_loader import Config
from src.utils.logger import get_logger


class YOLODetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "yolo_detector",
            cfg.get("logging", "log_path", default="logs/prime.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.model_path = cfg.get("yolo", "model_path", default="models/yolo/finetuned/best.pt")
        self.conf_threshold = cfg.get("yolo", "confidence_threshold", default=0.35)
        self.iou_threshold = cfg.get("yolo", "iou_threshold", default=0.45)
        self.input_size = cfg.get("yolo", "input_size", default=640)

        self.model = None
        self._load_model()

    def _load_model(self):
        if not Path(self.model_path).exists():
            self.logger.warning(
                f"YOLO weights not found at {self.model_path}"
            )

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.logger.info(f"YOLOv8 loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"YOLO load failed: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> list:
        """
        Run YOLOv8 on the full frame.
        Returns list of detections in frame coordinates.

        Returns:
            detections: list of dicts
                {x1, y1, x2, y2, confidence, class_id, class_name}
        """
        if self.model is None:
            self.logger.warning("YOLO model not loaded — skipping detection")
            return []

        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            verbose=False
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names.get(cls_id, "unknown")

                detections.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": cls_name
                })

        if detections:
            self.logger.debug(f"YOLO: {len(detections)} detection(s)")

        return detections

    def visualise(self, frame: np.ndarray, detections: list) -> np.ndarray:
        vis = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            label = f"FOD {det['confidence']:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return vis

    def __repr__(self):
        return f"YOLODetector(model={self.model_path}, conf={self.conf_threshold})"
