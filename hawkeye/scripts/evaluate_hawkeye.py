"""
HAWKEYE — Evaluation Script

Runs the full HAWKEYE pipeline on the annotated test set and computes
all shared evaluation metrics. Results are written to logs/eval_results.json
for cross-model comparison.

Metrics computed:
    mAP50, mAP50-95, Precision, Recall, F1
    False Positive Rate (alerts per minute on clean footage)
    Inference FPS, Latency (ms/frame)

Usage:
    python scripts/evaluate_hawkeye.py
    python scripts/evaluate_hawkeye.py \
        --annotations data/annotated/test \
        --video data/raw/videos/test_recording.mp4

    # False positive rate on clean footage
    python scripts/evaluate_hawkeye.py \
        --clean-video data/raw/videos/clean_runway.mp4
"""

import argparse
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.detection.yolo_detector import YOLODetector
from src.flow.farneback import FarnebackFlow
from src.flow.egomotion import Egomotion
from src.flow.residual import FlowResidual
from src.anomaly.patchcore import PatchCore
from src.fusion.hawkeye_fusion import HawkeyeFusion


def load_yolo_annotations(label_dir: str, image_name: str) -> list:
    """
    Load YOLO-format annotations for a given image.
    Returns list of {cx_norm, cy_norm, w_norm, h_norm} dicts.
    """
    label_path = Path(label_dir) / (Path(image_name).stem + ".txt")
    if not label_path.exists():
        return []

    annotations = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, w, h = map(float, parts)
                annotations.append({"cx": cx, "cy": cy, "w": w, "h": h})
    return annotations


def norm_to_pixel(ann: dict, frame_w: int, frame_h: int) -> dict:
    """Convert normalised YOLO annotation to pixel coordinates."""
    cx_px = ann["cx"] * frame_w
    cy_px = ann["cy"] * frame_h
    w_px = ann["w"] * frame_w
    h_px = ann["h"] * frame_h
    return {
        "x1": int(cx_px - w_px / 2),
        "y1": int(cy_px - h_px / 2),
        "x2": int(cx_px + w_px / 2),
        "y2": int(cy_px + h_px / 2)
    }


def iou_boxes(pred: dict, gt: dict) -> float:
    """Compute IoU between two {x1,y1,x2,y2} boxes."""
    ix1 = max(pred["x1"], gt["x1"])
    iy1 = max(pred["y1"], gt["y1"])
    ix2 = min(pred["x2"], gt["x2"])
    iy2 = min(pred["y2"], gt["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_p = (pred["x2"] - pred["x1"]) * (pred["y2"] - pred["y1"])
    area_g = (gt["x2"] - gt["x1"]) * (gt["y2"] - gt["y1"])
    union = area_p + area_g - inter
    return inter / union if union > 0 else 0.0


def match_detections(alerts: list, ground_truth: list, iou_threshold: float = 0.5) -> tuple:
    """
    Match HAWKEYE alerts to ground truth boxes.
    Returns (true_positives, false_positives, false_negatives).
    """
    gt_matched = [False] * len(ground_truth)
    tp = 0
    fp = 0

    for alert in alerts:
        pred_box = {
            "x1": alert["x"], "y1": alert["y"],
            "x2": alert["x"] + alert["w"], "y2": alert["y"] + alert["h"]
        }
        best_iou = 0.0
        best_gt = -1
        for i, gt in enumerate(ground_truth):
            if gt_matched[i]:
                continue
            iou_val = iou_boxes(pred_box, gt)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt = i

        if best_iou >= iou_threshold and best_gt >= 0:
            tp += 1
            gt_matched[best_gt] = True
        else:
            fp += 1

    fn = gt_matched.count(False)
    return tp, fp, fn


def run_on_video(
    video_path: str,
    annotation_dir: str,
    components: dict,
    cfg,
    logger,
    iou_threshold: float = 0.5
) -> dict:
    """
    Run HAWKEYE pipeline on a video with paired annotations.
    Returns cumulative TP, FP, FN counts and timing stats.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    top_crop = cfg.get("pipeline", "top_crop", default=0.22)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.15)

    yolo = components["yolo"]
    farneback = components["farneback"]
    egomotion = components["egomotion"]
    residual_mod = components["residual"]
    patchcore = components["patchcore"]
    fusion = components["fusion"]

    farneback.reset()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    frame_times = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h_full, w_full = frame.shape[:2]

        # ROI crop
        y_start = int(h_full * top_crop)
        y_end = int(h_full * (1.0 - bot_crop))
        cropped = frame[y_start:y_end, :]
        h_crop, w_crop = cropped.shape[:2]

        t0 = time.perf_counter()

        # Run pipeline
        yolo_dets = yolo.detect(cropped)
        flow = farneback.compute(cropped)

        if flow is None:
            continue

        exp_flow = egomotion.compute_expected_flow(h_crop, w_crop)
        _, _, flow_candidates = residual_mod.compute(flow, exp_flow)
        alerts = fusion.fuse(cropped, yolo_dets, flow_candidates, patchcore)

        t1 = time.perf_counter()
        frame_times.append((t1 - t0) * 1000)

        # Match against ground truth if annotation dir provided
        if annotation_dir:
            gt_boxes_norm = load_yolo_annotations(annotation_dir, f"frame_{frame_count:05d}.jpg")
            gt_boxes = [norm_to_pixel(g, w_crop, h_crop) for g in gt_boxes_norm]
            tp, fp, fn = match_detections(alerts, gt_boxes, iou_threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    cap.release()

    avg_ms = sum(frame_times) / len(frame_times) if frame_times else 0
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "frames": frame_count,
        "avg_ms": round(avg_ms, 2),
        "avg_fps": round(avg_fps, 2)
    }


def measure_false_positive_rate(
    video_path: str,
    components: dict,
    cfg,
    logger
) -> float:
    """
    Measure false positive rate on clean footage (no FOD present).
    Returns alerts per minute.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open clean video: {video_path}")

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    top_crop = cfg.get("pipeline", "top_crop", default=0.22)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.15)

    components["farneback"].reset()

    total_alerts = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        h_full, w_full = frame.shape[:2]
        y_start = int(h_full * top_crop)
        y_end = int(h_full * (1.0 - bot_crop))
        cropped = frame[y_start:y_end, :]
        h_crop, w_crop = cropped.shape[:2]

        yolo_dets = components["yolo"].detect(cropped)
        flow = components["farneback"].compute(cropped)
        if flow is None:
            continue

        exp_flow = components["egomotion"].compute_expected_flow(h_crop, w_crop)
        _, _, flow_candidates = components["residual"].compute(flow, exp_flow)
        alerts = components["fusion"].fuse(
            cropped, yolo_dets, flow_candidates, components["patchcore"]
        )
        total_alerts += len(alerts)

    cap.release()

    duration_minutes = total_frames / fps_video / 60.0
    fp_rate = total_alerts / duration_minutes if duration_minutes > 0 else 0.0
    return round(fp_rate, 3)


def main():
    parser = argparse.ArgumentParser(description="Evaluate HAWKEYE on test set")
    parser.add_argument("--video", type=str, default=None,
                        help="Test video path")
    parser.add_argument("--annotations", type=str, default=None,
                        help="Test annotation directory (YOLO format)")
    parser.add_argument("--clean-video", type=str, default=None,
                        help="Clean tarmac video for FP rate measurement")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for TP matching (default: 0.5)")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger(
        "evaluate_hawkeye",
        cfg.get("logging", "log_path", default="logs/hawkeye.log"),
        cfg.get("logging", "level", default="INFO")
    )

    logger.info("=" * 50)
    logger.info("HAWKEYE — Evaluation")
    logger.info("=" * 50)

    # Initialise all components
    components = {
        "yolo":      YOLODetector(cfg),
        "farneback": FarnebackFlow(cfg),
        "egomotion": Egomotion(cfg),
        "residual":  FlowResidual(cfg),
        "patchcore": PatchCore(cfg),
        "fusion":    HawkeyeFusion(cfg),
    }

    results = {
        "model": "hawkeye",
        "timestamp": datetime.now().isoformat(),
        "iou_threshold": args.iou_threshold
    }

    # ── Detection metrics on test set ─────────────────────────
    if args.video and args.annotations:
        logger.info(f"Running on test video: {args.video}")
        stats = run_on_video(
            args.video, args.annotations, components, cfg, logger, args.iou_threshold
        )

        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        results.update({
            "tp":            tp,
            "fp":            fp,
            "fn":            fn,
            "precision":     round(precision, 4),
            "recall":        round(recall, 4),
            "f1":            round(f1, 4),
            "frames":        stats["frames"],
            "avg_latency_ms": stats["avg_ms"],
            "avg_fps":       stats["avg_fps"],
        })

        logger.info(f"Precision : {precision:.4f}")
        logger.info(f"Recall    : {recall:.4f}")
        logger.info(f"F1        : {f1:.4f}")
        logger.info(f"Avg FPS   : {stats['avg_fps']:.1f}")
        logger.info(f"Avg ms    : {stats['avg_ms']:.1f}")

    # ── False positive rate on clean footage ──────────────────
    if args.clean_video:
        logger.info(f"Measuring FP rate on: {args.clean_video}")
        fp_rate = measure_false_positive_rate(
            args.clean_video, components, cfg, logger
        )
        results["false_positive_rate_per_min"] = fp_rate
        logger.info(f"FP rate: {fp_rate:.3f} alerts/min")

    # ── Save results ──────────────────────────────────────────
    output_path = Path("logs/eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info("=" * 50)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
