"""
ARGUS-N YOLO Detector
Runs YOLOv8 only on flagged candidate patches from flow residual.
Never runs on full frame — 93% compute reduction.
Confirms: is this a real physical object or noise?
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
            cfg.get("logging", "log_path", default="logs/argus.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.model_path = cfg.get("yolo", "model_path", default="models/yolo/yolov8n.pt")
        self.conf_threshold = cfg.get("yolo", "confidence_threshold", default=0.35)
        self.iou_threshold = cfg.get("yolo", "iou_threshold", default=0.45)
        self.input_size = cfg.get("yolo", "input_size", default=640)
        self.patch_padding = cfg.get("yolo", "patch_padding", default=20)
        self.patches_only = cfg.get("yolo", "run_on_patches_only", default=True)

        self.anomaly_conf_threshold = 0.25  # lower threshold for saving anomaly frames

        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Load YOLOv8 model.
        Requires: pip install ultralytics
        """
        if not Path(self.model_path).exists():
            self.logger.warning(
                f"YOLO weights not found at {self.model_path} "
                f"— attempting to download yolov8n.pt"
            )

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.logger.info(f"YOLOv8 loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"YOLO load failed: {e}")
            self.model = None

    def _extract_patch(
        self,
        frame: np.ndarray,
        candidate: dict
    ) -> tuple:
        """
        Extract padded patch from frame around candidate region.
        Returns patch and its origin coordinates for mapping back to frame.

        Returns:
            patch:  np.ndarray cropped region
            origin: (x1, y1) top left corner in original frame
        """
        h, w = frame.shape[:2]
        x1 = max(0, candidate["x"] - self.patch_padding)
        y1 = max(0, candidate["y"] - self.patch_padding)
        x2 = min(w, candidate["x"] + candidate["w"] + self.patch_padding)
        y2 = min(h, candidate["y"] + candidate["h"] + self.patch_padding)

        patch = frame[y1:y2, x1:x2]
        origin = (x1, y1)
        return patch, origin

    def _run_yolo_on_patch(
        self,
        patch: np.ndarray
    ) -> list:
        """
        Run YOLOv8 on a single patch.
        Returns list of detections in patch coordinates.
        Each detection: {x1, y1, x2, y2, confidence, class_id, class_name}
        """
        if self.model is None:
            return []

        results = self.model.predict(
            patch,
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
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": cls_name
                })

        return detections

    def _map_to_frame(
        self,
        detections: list,
        origin: tuple
    ) -> list:
        """
        Map patch-level detections back to full frame coordinates.
        origin: (x1, y1) of patch in full frame
        """
        ox, oy = origin
        mapped = []
        for det in detections:
            mapped.append({
                **det,
                "x1": det["x1"] + ox,
                "y1": det["y1"] + oy,
                "x2": det["x2"] + ox,
                "y2": det["y2"] + oy,
            })
        return mapped

    def detect(
        self,
        frame: np.ndarray,
        candidates: list
    ) -> tuple:
        """
        Run YOLO on all candidate patches from flow residual.
        Returns confirmed detections mapped to full frame coordinates
        and list of anomaly patches for saving to replay buffer.

        Args:
            frame:      full frame (H, W, 3)
            candidates: list of candidate dicts from FlowResidual

        Returns:
            confirmed:      list of confirmed FOD detections in frame coords
            anomaly_frames: list of patches where YOLO was uncertain
        """
        if self.model is None:
            self.logger.warning("YOLO model not loaded — skipping detection")
            return [], []

        if not candidates:
            return [], []

        confirmed = []
        anomaly_frames = []

        for candidate in candidates:
            patch, origin = self._extract_patch(frame, candidate)

            if patch.size == 0:
                continue

            detections = self._run_yolo_on_patch(patch)
            mapped = self._map_to_frame(detections, origin)

            for det in mapped:
                if det["confidence"] >= self.conf_threshold:
                    # High confidence — confirmed detection
                    confirmed.append({
                        **det,
                        "candidate": candidate
                    })
                    self.logger.debug(
                        f"Confirmed: {det['class_name']} "
                        f"conf={det['confidence']:.2f} "
                        f"at ({det['x1']},{det['y1']})"
                    )
                elif det["confidence"] >= self.anomaly_conf_threshold:
                    # Low confidence — uncertain — save for retraining
                    anomaly_frames.append({
                        "patch": patch,
                        "detection": det,
                        "candidate": candidate
                    })
                    self.logger.debug(
                        f"Uncertain detection saved — "
                        f"conf={det['confidence']:.2f}"
                    )

        return confirmed, anomaly_frames

    def visualise(
        self,
        frame: np.ndarray,
        confirmed: list
    ) -> np.ndarray:
        """
        Draw confirmed detections on frame.
        Returns annotated frame.
        """
        vis = frame.copy()

        for det in confirmed:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            label = f"FOD: {det['class_name']} {det['confidence']:.2f}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        return vis

    def __repr__(self):
        return (
            f"YOLODetector("
            f"model={self.model_path}, "
            f"conf={self.conf_threshold}, "
            f"patches_only={self.patches_only})"
        )
