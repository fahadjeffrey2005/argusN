"""
evaluate_prime.py — Evaluate PRIME on the held-out test set.

Metrics (identical schema to yolofinetune and hawkeye for direct comparison):
  - mAP50, mAP50-95, Precision, Recall, F1  (YOLO val on test split)
  - False Positive Rate  (full PRIME pipeline on clean tarmac video EVAL SEGMENT)
  - Inference FPS and latency ms/frame

IMPORTANT — clean video hard split:
    The FP evaluation ONLY uses frames >= CLEAN_EVAL_START_FRAME (frame 6300,
    i.e. the 210s mark at 30fps).  Frames 0..6299 were used for crop collection
    and CNN training — using them here would contaminate the FP metric.

Usage (run from inside prime/ on Ubuntu):
    python scripts/evaluate_prime.py \\
        --data config/dataset.yaml \\
        --clean-video ../yolofinetune/data/raw/videos/clean_runway.mp4 \\
        --device cuda
"""

import cv2
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.detection.yolo_detector import YOLODetector
from src.flow.frame_diff import FrameDiff
from src.semantic.crop_builder import CropBuilder
from src.semantic.cnn_classifier import CNNClassifier
from src.tracking.temporal_tracker import TemporalTracker

# Frames 0..6299 → CNN training.  Frames 6300+ → FP evaluation only.
CLEAN_EVAL_START_FRAME = 6300


def apply_roi(frame, top_frac, bot_frac):
    h, w = frame.shape[:2]
    y0 = int(h * top_frac)
    y1 = int(h * (1.0 - bot_frac))
    return frame[y0:y1, :], y0


def evaluate_on_dataset(model, data_yaml: Path, imgsz: int, device: str, log) -> dict:
    """YOLO val() on test split — mAP50, mAP50-95, precision, recall."""
    log.info(f"Running YOLO val() on {data_yaml} (test split)...")
    metrics = model.val(data=str(data_yaml), split="test",
                        imgsz=imgsz, device=device, verbose=False)
    p  = float(metrics.box.mp)
    r  = float(metrics.box.mr)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    results = {
        "mAP50":     float(metrics.box.map50),
        "mAP50_95":  float(metrics.box.map),
        "precision": round(p,  4),
        "recall":    round(r,  4),
        "f1":        round(f1, 4),
    }
    log.info(f"mAP50     : {results['mAP50']:.4f}")
    log.info(f"mAP50-95  : {results['mAP50_95']:.4f}")
    log.info(f"Precision : {results['precision']:.4f}")
    log.info(f"Recall    : {results['recall']:.4f}")
    log.info(f"F1        : {results['f1']:.4f}")
    return results


def benchmark_fps(cfg, log, n_frames: int = 300) -> dict:
    """
    Benchmark the full PRIME pipeline on synthetic tarmac-coloured frames.
    Uses FrameDiff (not Farneback) — matches the real pipeline.
    """
    from ultralytics import YOLO

    log.info(f"FPS benchmark ({n_frames} synthetic frames)...")

    top_crop = cfg.get("pipeline", "top_crop", default=0.50)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.05)
    imgsz    = cfg.get("yolo",     "input_size", default=640)
    conf_t   = cfg.get("yolo",     "confidence_threshold", default=0.28)
    iou_t    = cfg.get("yolo",     "iou_threshold", default=0.45)
    device   = cfg.device

    model      = YOLO(cfg.get("yolo", "model_path", default="models/yolo/finetuned/best.pt"))
    diff       = FrameDiff()
    crop_bld   = CropBuilder(cfg)
    classifier = CNNClassifier(cfg)
    prefilter  = TemporalTracker(cfg, confirm_frames_override=2)
    tracker    = TemporalTracker(cfg)

    H = int(1080 * (1.0 - top_crop - bot_crop))
    latencies = []

    for _ in range(n_frames):
        frame = np.random.randint(80, 180, (H, 1920, 3), dtype=np.uint8)
        t0 = time.perf_counter()

        results = model.predict(frame, imgsz=imgsz, conf=conf_t,
                                iou=iou_t, verbose=False, device=device)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append({"x1": int(x1), "y1": int(y1),
                             "x2": int(x2), "y2": int(y2),
                             "confidence": float(box.conf[0])})

        diff_mag = diff.compute(frame)
        crops    = crop_bld.build_batch(frame, diff_mag, dets)
        clf_res  = classifier.classify_batch(crops)

        fast_cands = []
        slow_cands = []
        for r, c in clf_res:
            if r["is_fod"]:
                det = {"x1": c["x1"], "y1": c["y1"],
                       "x2": c["x2"], "y2": c["y2"],
                       "confidence": r["confidence"]}
                if r["fast_path"]:
                    fast_cands.append(det)
                else:
                    slow_cands.append(det)

        prefilter_confirmed = prefilter.update(slow_cands)
        tracker.update(fast_cands + prefilter_confirmed)

        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    fps    = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    log.info(f"Latency   : {avg_ms:.1f} ms/frame")
    log.info(f"FPS       : {fps:.1f}")
    return {"latency_ms": round(avg_ms, 2), "fps": round(fps, 1)}


def evaluate_false_positive_rate(cfg, video_path: str, log) -> dict:
    """
    Run full PRIME pipeline on the EVAL SEGMENT of the clean tarmac video.

    ONLY processes frames >= CLEAN_EVAL_START_FRAME (frame 6300 = 210s @ 30fps).
    Frames before that were used for CNN training — using them here would
    contaminate the false positive metric.
    """
    from ultralytics import YOLO

    vp = Path(video_path)
    if not vp.exists():
        log.warning(f"Clean video not found: {vp}")
        return {"fp_per_minute": None, "fp_total": None,
                "clean_video_duration_s": None, "eval_start_frame": CLEAN_EVAL_START_FRAME}

    top_crop = cfg.get("pipeline", "top_crop", default=0.50)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.05)
    imgsz    = cfg.get("yolo",     "input_size", default=640)
    conf_t   = cfg.get("yolo",     "confidence_threshold", default=0.28)
    iou_t    = cfg.get("yolo",     "iou_threshold", default=0.45)
    device   = cfg.device
    confirm  = cfg.get("tracker",  "confirm_frames", default=3)

    model      = YOLO(cfg.get("yolo", "model_path", default="models/yolo/finetuned/best.pt"))
    diff       = FrameDiff()
    crop_bld   = CropBuilder(cfg)
    classifier = CNNClassifier(cfg)
    prefilter  = TemporalTracker(cfg, confirm_frames_override=2)
    tracker    = TemporalTracker(cfg)

    cap          = cv2.VideoCapture(str(vp))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log.info(f"Clean video : {vp.name} — {total_frames} frames ({total_frames/src_fps:.1f}s)")
    log.info(f"Eval segment: frames {CLEAN_EVAL_START_FRAME}..{total_frames} "
             f"({(total_frames - CLEAN_EVAL_START_FRAME)/src_fps:.1f}s)")

    # Seek to eval start frame (also keeps FrameDiff warm with prior frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, CLEAN_EVAL_START_FRAME - 1))
    # Prime FrameDiff with one frame before the eval window
    ret, warmup_frame = cap.read()
    if ret and warmup_frame is not None:
        h = warmup_frame.shape[0]
        warmup_roi = warmup_frame[int(h * top_crop):int(h * (1.0 - bot_crop)), :]
        diff.compute(warmup_roi)

    eval_frames = total_frames - CLEAN_EVAL_START_FRAME
    duration_s  = eval_frames / src_fps
    fp_events   = 0
    frame_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        h   = frame.shape[0]
        roi = frame[int(h * top_crop):int(h * (1.0 - bot_crop)), :]
        roi_h, roi_w = roi.shape[:2]

        diff_mag = diff.compute(roi)

        results = model.predict(roi, imgsz=imgsz, conf=conf_t,
                                iou=iou_t, verbose=False, device=device)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append({"x1": int(x1), "y1": int(y1),
                             "x2": int(x2), "y2": int(y2),
                             "confidence": float(box.conf[0])})

        crops   = crop_bld.build_batch(roi, diff_mag, dets)
        clf_res = classifier.classify_batch(crops)

        fast_cands = []
        slow_cands = []
        for r, c in clf_res:
            if r["is_fod"]:
                det = {"x1": c["x1"], "y1": c["y1"],
                       "x2": c["x2"], "y2": c["y2"],
                       "confidence": r["confidence"]}
                if r["fast_path"]:
                    fast_cands.append(det)
                else:
                    slow_cands.append(det)

        prefilter_confirmed = prefilter.update(slow_cands)
        tracker.update(fast_cands + prefilter_confirmed)

        # Count first confirmation frame as one FP event
        for t in tracker.tracks:
            if t.confirmed and t.hits == confirm:
                fp_events += 1

        if frame_idx % 500 == 0:
            log.info(f"  {frame_idx}/{eval_frames} eval frames — FP events: {fp_events}")

    cap.release()
    fp_per_min = (fp_events / duration_s * 60) if duration_s > 0 else 0.0
    log.info(f"FP total  : {fp_events}")
    log.info(f"Eval duration : {duration_s:.1f}s")
    log.info(f"FP rate   : {fp_per_min:.3f}/min")

    return {
        "fp_per_minute":           round(fp_per_min, 3),
        "fp_total":                fp_events,
        "clean_video_duration_s":  round(duration_s, 1),
        "eval_start_frame":        CLEAN_EVAL_START_FRAME,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PRIME.")
    parser.add_argument("--model",       default="models/yolo/finetuned/best.pt")
    parser.add_argument("--data",        default="config/dataset.yaml")
    parser.add_argument("--clean-video", default=None)
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--device",      default=None)
    parser.add_argument("--output",      default="logs/eval_results.json")
    args = parser.parse_args()

    from ultralytics import YOLO

    cfg = load_config(args.config)
    log = get_logger("evaluate_prime",
                     cfg.get("logging", "log_path", default="logs/prime.log"),
                     cfg.get("logging", "level",    default="INFO"))

    if args.device:
        cfg._cfg["device"] = args.device

    model_path = Path(args.model)
    if not model_path.exists():
        log.error(f"YOLO weights not found: {model_path}")
        sys.exit(1)

    log.info("=" * 55)
    log.info("PRIME — Evaluation")
    log.info("=" * 55)
    log.info(f"Model  : {model_path}")
    log.info(f"Device : {cfg.device}")

    model   = YOLO(str(model_path))
    results = {"model": "prime", "model_path": str(model_path)}

    # mAP metrics via YOLO val() on test split
    data_yaml = Path(args.data)
    if data_yaml.exists():
        results.update(evaluate_on_dataset(
            model, data_yaml,
            cfg.get("yolo", "input_size", default=640),
            cfg.device, log
        ))
    else:
        log.warning(f"Dataset YAML not found: {data_yaml} — skipping mAP evaluation")

    # FPS benchmark
    log.info("\nBenchmarking pipeline speed...")
    results.update(benchmark_fps(cfg, log))

    # False positive rate — eval segment only
    if args.clean_video:
        log.info(f"\nMeasuring FP rate on eval segment "
                 f"(frames {CLEAN_EVAL_START_FRAME}+)...")
        results.update(evaluate_false_positive_rate(cfg, args.clean_video, log))

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nResults saved → {out}")
    log.info("\n── Summary ──")
    for k, v in results.items():
        if k not in ("model", "model_path"):
            log.info(f"  {k:<36}: {v}")


if __name__ == "__main__":
    main()
