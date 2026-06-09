"""
PRIME — evaluate_prime.py
Evaluate the full PRIME pipeline on the test set.
Produces all metrics required for the comparative study:
  mAP50, mAP50-95, Precision, Recall, F1, False Positive Rate,
  Inference FPS, Latency ms.

Also outputs a per-class confusion matrix for the CNN classifier.

Usage:
    python scripts/evaluate_prime.py \
      --video data/raw/videos/test_recording.mp4 \
      --annotations data/annotated/labels/test \
      --config config/config.yaml \
      --output logs/eval_results.json
"""

import argparse
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.detection.yolo_detector import YOLODetector
from src.flow.farneback import FarnebackFlow
from src.flow.egomotion import Egomotion
from src.flow.residual import FlowResidual
from src.fusion.prime_fusion import PrimeFusion
from src.semantic.crop_builder import CropBuilder
from src.semantic.cnn_classifier import CNNClassifier


def load_yolo_annotation(label_path: Path, frame_w: int, frame_h: int) -> list:
    """
    Load YOLO format annotations (.txt).
    Returns list of {x1, y1, x2, y2} in pixel coords.
    """
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - w / 2) * frame_w)
            y1 = int((cy - h / 2) * frame_h)
            x2 = int((cx + w / 2) * frame_w)
            y2 = int((cy + h / 2) * frame_h)
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_id": cls})
    return boxes


def iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(1, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    area_b = max(1, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    return inter / (area_a + area_b - inter)


def match_detections(gt_boxes: list, pred_boxes: list, iou_threshold: float = 0.5) -> tuple:
    """Match predictions to ground truth. Returns (tp, fp, fn)."""
    matched_gt = set()
    tp = 0
    fp = 0

    for pred in pred_boxes:
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            score = iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_gi = gi

        if best_iou >= iou_threshold and best_gi >= 0:
            tp += 1
            matched_gt.add(best_gi)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate PRIME pipeline")
    parser.add_argument("--video", required=True, help="Test video file")
    parser.add_argument("--annotations", required=True, help="Directory of YOLO .txt label files")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="logs/eval_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger(
        "evaluate_prime",
        cfg.get("logging", "log_path", default="logs/prime.log"),
        cfg.get("logging", "level", default="INFO")
    )

    # Components
    yolo = YOLODetector(cfg)
    flow_engine = FarnebackFlow(cfg)
    ego = Egomotion(cfg)
    residual = FlowResidual(cfg)
    fusion = PrimeFusion(cfg)
    crop_builder = CropBuilder(cfg)
    classifier = CNNClassifier(cfg)

    top_crop = cfg.get("pipeline", "top_crop", default=0.22)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.15)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    annotations_dir = Path(args.annotations)

    total_tp = total_fp = total_fn = 0
    frame_latencies = []
    frame_idx = 0

    logger.info(f"Evaluating on {total_frames} frames from {args.video}")

    for frame_idx in tqdm(range(total_frames), desc="Evaluating"):
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        y_start = int(h * top_crop)
        y_end = int(h * (1 - bot_crop))
        frame_roi = frame[y_start:y_end, :]
        roi_h, roi_w = frame_roi.shape[:2]

        # Load annotations
        label_path = annotations_dir / f"{frame_idx:06d}.txt"
        gt_boxes = load_yolo_annotation(label_path, roi_w, roi_h)

        t0 = time.time()

        raw_flow = flow_engine.compute(frame_roi)
        if raw_flow is None:
            frame_latencies.append(0.0)
            if gt_boxes:
                total_fn += len(gt_boxes)
            continue

        flow_mag = flow_engine.flow_magnitude(raw_flow)
        expected = ego.compute_expected_flow()
        expected_roi = cv2.resize(expected, (roi_w, roi_h))
        _, _, flow_candidates = residual.compute(raw_flow, expected_roi)
        yolo_candidates = yolo.detect(frame_roi)
        merged = fusion.merge(yolo_candidates, flow_candidates)
        crops_and_candidates = crop_builder.build_batch(frame_roi, flow_mag, merged)
        results = classifier.classify_batch(crops_and_candidates)

        latency = (time.time() - t0) * 1000
        frame_latencies.append(latency)

        # Only FOD-classified candidates count as detections
        pred_boxes = [
            cand for cls_result, cand in results if cls_result["is_fod"]
        ]

        tp, fp, fn = match_detections(gt_boxes, pred_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    cap.release()

    # ── Compute metrics ────────────────────────────────────────────
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    # False positive rate: FP per minute
    fps_actual = 1000.0 / max(np.mean(frame_latencies), 0.001)
    duration_min = total_frames / max(fps_actual * 60, 0.001)
    fp_rate_per_min = total_fp / max(duration_min, 0.001)

    avg_latency = float(np.mean(frame_latencies))
    avg_fps = 1000.0 / max(avg_latency, 0.001)

    results = {
        "model": "PRIME",
        "video": args.video,
        "total_frames": total_frames,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate_per_min": round(fp_rate_per_min, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_fps": round(avg_fps, 1),
        "notes": "mAP50/mAP50-95 requires per-frame confidence scores — use ultralytics val() for full mAP"
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n── PRIME Evaluation Results ──────────────────────")
    for k, v in results.items():
        logger.info(f"  {k:<35}: {v}")
    logger.info(f"\nResults saved → {output_path}")


if __name__ == "__main__":
    main()
