"""
enhanced/src/detection/detector.py
------------------------------------
YOLO detector with optional SAHI-style sliced inference.

Why sliced inference matters for tiny FODs:
  A 5mm FOD at typical runway camera distance is ~10px in a 1920×1080 frame.
  Without slicing: the ROI (e.g. 1920×540) is squashed to 640×640 → FOD becomes 3px.
  YOLO cannot reliably detect a 3-pixel object.

  With slicing: the ROI is divided into overlapping 640×640 patches at native resolution.
  That same 10px FOD stays 10px in its patch → YOLO can see it.

Config options (enhanced_config.yaml):
  model:
    use_sliced: true     # enable sliced inference (recommended for tiny FODs)
    slice_size: 640      # patch size (must match model training imgsz)
    slice_overlap: 0.2   # fractional overlap between adjacent slices (0.0-0.5)
"""

from dataclasses import dataclass
from typing import List, Tuple
import math
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

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def as_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "conf": self.conf, "cls": self.cls}


def _iou(a: Detection, b: Detection) -> float:
    """Intersection over Union for NMS."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def _nms(dets: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """
    Simple greedy NMS to merge duplicate detections from overlapping slices.
    Keeps highest-conf box when two boxes overlap more than iou_threshold.
    """
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    kept = []
    suppressed = [False] * len(dets)
    for i, d in enumerate(dets):
        if suppressed[i]:
            continue
        kept.append(d)
        for j in range(i + 1, len(dets)):
            if not suppressed[j] and _iou(d, dets[j]) > iou_threshold:
                suppressed[j] = True
    return kept


def _slice_coords(length: int, slice_size: int, overlap: float) -> List[Tuple[int, int]]:
    """
    Compute (start, end) positions for slicing a dimension of `length`.
    If length <= slice_size, returns a single (0, length) slice.
    """
    if length <= slice_size:
        return [(0, length)]
    stride = int(slice_size * (1.0 - overlap))
    stride = max(1, stride)
    coords = []
    start = 0
    while start < length:
        end = min(start + slice_size, length)
        coords.append((start, end))
        if end == length:
            break
        start += stride
    return coords


class EnhancedDetector:
    def __init__(self, model_path: str, conf: float, imgsz: int, device: str,
                 top_crop: float, bot_crop: float,
                 left_crop: float = 0.0, right_crop: float = 0.0,
                 use_sliced: bool = True,
                 slice_size: int = 640,
                 slice_overlap: float = 0.2):
        from ultralytics import YOLO
        self.model         = YOLO(model_path)
        self.conf          = conf
        self.imgsz         = imgsz
        self.device        = device
        self.top_crop      = top_crop
        self.bot_crop      = bot_crop
        self.left_crop     = left_crop
        self.right_crop    = right_crop
        self.use_sliced    = use_sliced
        self.slice_size    = slice_size
        self.slice_overlap = slice_overlap
        self.class_names: dict = self.model.names

    def roi_bounds(self, h: int, w: int):
        y_start = int(h * self.top_crop)
        y_end   = int(h * (1.0 - self.bot_crop))
        x_start = int(w * self.left_crop)
        x_end   = int(w * (1.0 - self.right_crop))
        return y_start, y_end, x_start, x_end

    def _detect_patch(self, patch: np.ndarray, offset_x: int, offset_y: int) -> List[Detection]:
        """Run YOLO on a single patch, return detections in full-frame coordinates."""
        results = self.model.predict(
            patch, conf=self.conf, imgsz=self.slice_size,
            verbose=False, device=self.device
        )
        dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                dets.append(Detection(
                    x1=x1 + offset_x, y1=y1 + offset_y,
                    x2=x2 + offset_x, y2=y2 + offset_y,
                    conf=float(box.conf[0]),
                    cls=int(box.cls[0]),
                ))
        return dets

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        y_start, y_end, x_start, x_end = self.roi_bounds(h, w)
        roi = frame[y_start:y_end, x_start:x_end]
        roi_h, roi_w = roi.shape[:2]

        if not self.use_sliced:
            # Original single-pass mode (fast but misses tiny objects)
            results = self.model.predict(roi, conf=self.conf, imgsz=self.imgsz,
                                          verbose=False, device=self.device)
            dets = []
            for r in results:
                for box in r.boxes:
                    cx1, cy1, cx2, cy2 = [int(v) for v in box.xyxy[0].tolist()]
                    dets.append(Detection(
                        x1=cx1 + x_start, y1=cy1 + y_start,
                        x2=cx2 + x_start, y2=cy2 + y_start,
                        conf=float(box.conf[0]),
                        cls=int(box.cls[0]),
                    ))
            return dets

        # ── Sliced inference (SAHI-style) ────────────────────
        # Divide ROI into overlapping patches at NATIVE resolution.
        # A 5mm FOD stays at its real pixel size within each patch.
        y_slices = _slice_coords(roi_h, self.slice_size, self.slice_overlap)
        x_slices = _slice_coords(roi_w, self.slice_size, self.slice_overlap)

        all_dets = []
        for ys, ye in y_slices:
            for xs, xe in x_slices:
                patch = roi[ys:ye, xs:xe]
                ph, pw = patch.shape[:2]

                # Pad patch to slice_size if smaller (edge case)
                if ph < self.slice_size or pw < self.slice_size:
                    padded = np.zeros((self.slice_size, self.slice_size, 3), dtype=np.uint8)
                    padded[:ph, :pw] = patch
                    patch = padded

                # Full-frame offset for coordinate mapping
                off_x = x_start + xs
                off_y = y_start + ys
                patch_dets = self._detect_patch(patch, off_x, off_y)
                all_dets.extend(patch_dets)

        # NMS to remove duplicates from overlapping slice regions
        return _nms(all_dets, iou_threshold=0.5)
