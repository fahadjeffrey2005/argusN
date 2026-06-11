"""
evaluate_hawkeye.py — Evaluate HAWKEYE on the held-out test set.

Metrics (identical schema to yolofinetune for direct comparison):
  - mAP50, mAP50-95, Precision, Recall, F1  (YOLO val on test split)
  - False Positive Rate  (full HAWKEYE pipeline on clean tarmac video)
  - Inference FPS and latency ms/frame

Usage (run from inside hawkeye/ directory):
    python scripts/evaluate_hawkeye.py \\
        --data config/dataset.yaml \\
        --clean-video ../yolofinetune/data/raw/videos/clean1.mp4 \\
        --device cuda
"""

import cv2
import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.tracking.temporal_tracker import TemporalTracker


def apply_roi_crop(frame, top_frac, bot_frac):
    h, w = frame.shape[:2]
    y_start = int(h * top_frac)
    y_end   = int(h * (1.0 - bot_frac))
    return frame[y_start:y_end, :], y_start


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
        "precision": p,
        "recall":    r,
        "f1":        f1,
    }
    log.info(f"mAP50     : {results['mAP50']:.4f}")
    log.info(f"mAP50-95  : {results['mAP50_95']:.4f}")
    log.info(f"Precision : {results['precision']:.4f}")
    log.info(f"Recall    : {results['recall']:.4f}")
    log.info(f"F1        : {results['f1']:.4f}")
    return results


def benchmark_fps(model, cfg, log, n_frames: int = 300) -> dict:
    """
    Benchmark YOLO + TemporalTracker on synthetic frames.
    Represents real pipeline speed.
    """
    import numpy as np
    log.info(f"FPS benchmark ({n_frames} frames)...")

    tracker  = TemporalTracker(cfg)
    top_crop = cfg.get("pipeline", "top_crop", default=0.60)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.05)
    imgsz    = cfg.get("yolo", "input_size",   default=640)
    conf_t   = cfg.get("yolo", "confidence_threshold", default=0.35)
    iou_t    = cfg.get("yolo", "iou_threshold",         default=0.45)
    device   = cfg.device

    H  = int(1080 * (1.0 - top_crop - bot_crop))
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
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                dets.append({"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),
                              "confidence":float(box.conf[0]),"class_id":0,"class_name":"fod"})
        tracker.update(dets)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    fps    = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    log.info(f"Latency   : {avg_ms:.1f} ms/frame")
    log.info(f"FPS       : {fps:.1f}")
    return {"latency_ms": avg_ms, "fps": fps}


def evaluate_false_positive_rate(model, cfg, video_path: str, log) -> dict:
    """
    Run full HAWKEYE pipeline on clean tarmac video.
    A false positive is a track that reaches confirmation threshold
    with no real FOD present.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        log.warning(f"Clean video not found: {video_path}")
        return {"fp_per_minute": None, "fp_total": None, "clean_video_duration_s": None}

    top_crop = cfg.get("pipeline", "top_crop", default=0.60)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.05)
    imgsz    = cfg.get("yolo", "input_size",   default=640)
    conf_t   = cfg.get("yolo", "confidence_threshold", default=0.35)
    iou_t    = cfg.get("yolo", "iou_threshold",         default=0.45)
    device   = cfg.device

    tracker = TemporalTracker(cfg)
    cap     = cv2.VideoCapture(str(video_path))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / src_fps

    log.info(f"Clean video : {video_path.name} — {total_frames} frames ({duration_s:.1f}s)")

    fp_events  = 0
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        h, w = frame.shape[:2]
        cropped = frame[int(h * top_crop):int(h * (1.0 - bot_crop)), :]

        results = model.predict(cropped, imgsz=imgsz, conf=conf_t,
                                iou=iou_t, verbose=False, device=device)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                dets.append({"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),
                              "confidence":float(box.conf[0]),"class_id":0,"class_name":"fod"})

        confirmed = tracker.update(dets)
        # Count first frame a track gets confirmed as one FP event
        for t in tracker.tracks:
            if t.confirmed and t.hits == cfg.get("tracker","confirm_frames",default=4):
                fp_events += 1

        if frame_idx % 500 == 0:
            log.info(f"  {frame_idx}/{total_frames} frames — FP events so far: {fp_events}")

    cap.release()
    fp_per_min = (fp_events / duration_s * 60) if duration_s > 0 else 0.0
    log.info(f"FP total : {fp_events} confirmed false tracks on clean footage")
    log.info(f"FP rate  : {fp_per_min:.2f} per minute")
    return {"fp_per_minute": fp_per_min, "fp_total": fp_events,
            "clean_video_duration_s": duration_s}


def main():
    parser = argparse.ArgumentParser(description="Evaluate HAWKEYE.")
    parser.add_argument("--model",       default="models/yolo/finetuned/best.pt")
    parser.add_argument("--data",        default="config/dataset.yaml")
    parser.add_argument("--clean-video", default=None)
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--device",      default=None)
    parser.add_argument("--output",      default="logs/eval_results.json")
    args = parser.parse_args()

    from ultralytics import YOLO

    cfg = load_config(args.config)
    log = get_logger("evaluate_hawkeye",
                     cfg.get("logging", "log_path", default="logs/hawkeye.log"),
                     cfg.get("logging", "level",    default="INFO"))

    if args.device:
        cfg._cfg["device"] = args.device

    model_path = Path(args.model)
    if not model_path.exists():
        log.error(f"YOLO weights not found: {model_path}")
        sys.exit(1)

    log.info("=" * 55)
    log.info("HAWKEYE — Evaluation")
    log.info("=" * 55)
    log.info(f"Model  : {model_path}")
    log.info(f"Device : {cfg.device}")

    model   = YOLO(str(model_path))
    results = {"model": "hawkeye", "model_path": str(model_path)}

    # Detection metrics
    data_yaml = Path(args.data)
    if data_yaml.exists():
        results.update(evaluate_on_dataset(model, data_yaml,
                       cfg.get("yolo","input_size",default=640), cfg.device, log))
    else:
        log.warning(f"Dataset YAML not found: {data_yaml}")

    # FPS
    log.info("\nBenchmarking pipeline speed...")
    results.update(benchmark_fps(model, cfg, log))

    # FP rate
    if args.clean_video:
        log.info("\nMeasuring false positive rate...")
        results.update(evaluate_false_positive_rate(model, cfg, args.clean_video, log))

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nResults saved → {out}")
    log.info("\n── Summary ──")
    for k, v in results.items():
        if k not in ("model", "model_path"):
            log.info(f"  {k:<30}: {v}")


if __name__ == "__main__":
    main()
