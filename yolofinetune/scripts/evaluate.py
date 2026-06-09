"""
evaluate.py — Evaluate fine-tuned YOLOv8n on the held-out test set.

Computes all ARGUS-N standard metrics:
  - mAP50, mAP50-95
  - Precision, Recall, F1
  - False Positive Rate (false alerts per minute on clean video)
  - Inference FPS and latency ms/frame

Results saved to logs/eval_results.json — consumed by compare_all.py.

Usage:
    # Full evaluation on test set:
    python scripts/evaluate.py \
        --model models/yolo/finetuned/best.pt \
        --data config/dataset.yaml

    # FP rate on clean video:
    python scripts/evaluate.py \
        --model models/yolo/finetuned/best.pt \
        --clean-video data/raw/videos/clean_runway.mp4

    # Both:
    python scripts/evaluate.py \
        --model models/yolo/finetuned/best.pt \
        --data config/dataset.yaml \
        --clean-video data/raw/videos/clean_runway.mp4
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


def evaluate_on_dataset(model, data_yaml: Path, imgsz: int, device: str, log) -> dict:
    """
    Run YOLO validation on test split.
    Returns dict with mAP50, mAP50-95, precision, recall.
    """
    log.info(f"Running YOLO validation on {data_yaml}...")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    results = {
        "mAP50":      float(metrics.box.map50),
        "mAP50_95":   float(metrics.box.map),
        "precision":  float(metrics.box.mp),
        "recall":     float(metrics.box.mr),
    }
    f1 = 0.0
    p = results["precision"]
    r = results["recall"]
    if (p + r) > 0:
        f1 = 2 * p * r / (p + r)
    results["f1"] = f1

    log.info(f"mAP50     : {results['mAP50']:.4f}")
    log.info(f"mAP50-95  : {results['mAP50_95']:.4f}")
    log.info(f"Precision : {results['precision']:.4f}")
    log.info(f"Recall    : {results['recall']:.4f}")
    log.info(f"F1        : {results['f1']:.4f}")

    return results


def evaluate_fps(model, data_yaml: Path, imgsz: int, device: str, log, n_frames: int = 200) -> dict:
    """
    Measure inference speed on a sample of test images.
    Returns latency_ms and fps.
    """
    import yaml
    with open(data_yaml) as f:
        ds = yaml.safe_load(f)
    test_path = Path(ds.get("test", ""))
    images = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
    if not images:
        log.warning("No test images found for FPS benchmark")
        return {"latency_ms": 0.0, "fps": 0.0}

    images = images[:n_frames]
    log.info(f"FPS benchmark on {len(images)} images...")

    latencies = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        t0 = time.perf_counter()
        model(img, imgsz=imgsz, device=device, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    if not latencies:
        return {"latency_ms": 0.0, "fps": 0.0}

    avg_ms = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    log.info(f"Latency   : {avg_ms:.1f} ms/frame")
    log.info(f"FPS       : {fps:.1f}")

    return {"latency_ms": avg_ms, "fps": fps}


def evaluate_false_positive_rate(
    model, video_path: str, cfg, imgsz: int, device: str, log
) -> dict:
    """
    Run model on a clean tarmac video (no FODs present).
    Counts false alerts. Returns FP rate per minute.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        log.warning(f"Clean video not found: {video_path}")
        return {"fp_per_minute": None, "fp_total": None, "clean_video_duration_s": None}

    conf_thresh = cfg.get("yolo", "confidence_threshold", default=0.35)
    iou_thresh  = cfg.get("yolo", "iou_threshold", default=0.45)
    top_crop    = cfg.get("pipeline", "top_crop", default=0.22)
    bot_crop    = cfg.get("pipeline", "bot_crop", default=0.15)

    cap = cv2.VideoCapture(str(video_path))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / source_fps

    log.info(f"Clean video : {video_path.name} — {total_frames} frames ({duration_s:.1f}s)")

    fp_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        y_start = int(h * top_crop)
        y_end   = int(h * (1.0 - bot_crop))
        cropped = frame[y_start:y_end, :]

        results = model(cropped, imgsz=imgsz, conf=conf_thresh, iou=iou_thresh, verbose=False)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                fp_count += 1
                break  # one alert per frame max

        frame_idx += 1

    cap.release()

    fp_per_min = (fp_count / duration_s * 60) if duration_s > 0 else 0.0
    log.info(f"FP total    : {fp_count} alerts on {frame_idx} clean frames")
    log.info(f"FP rate     : {fp_per_min:.2f} per minute")

    return {
        "fp_per_minute": fp_per_min,
        "fp_total": fp_count,
        "clean_video_duration_s": duration_s
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOFINETUNE on test set.")
    parser.add_argument("--model", default="models/yolo/finetuned/best.pt")
    parser.add_argument("--data", default="config/dataset.yaml", help="Dataset YAML")
    parser.add_argument("--clean-video", default=None, help="Clean runway video for FP rate")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="logs/eval_results.json")
    args = parser.parse_args()

    from ultralytics import YOLO

    cfg = load_config(args.config)
    log = get_logger(
        "evaluate",
        cfg.get("logging", "log_path", default="logs/yolofinetune.log"),
        cfg.get("logging", "level", default="INFO")
    )

    device = args.device or cfg.device
    imgsz  = cfg.get("yolo", "input_size", default=640)

    model_path = Path(args.model)
    if not model_path.exists():
        log.error(f"Model not found: {model_path}")
        sys.exit(1)

    log.info("=" * 50)
    log.info("YOLOFINETUNE — Evaluation")
    log.info("=" * 50)
    log.info(f"Model  : {model_path}")
    log.info(f"Device : {device}")

    model = YOLO(str(model_path))

    results = {"model": "yolofinetune", "model_path": str(model_path)}

    # Detection metrics
    data_yaml = Path(args.data)
    if data_yaml.exists():
        det_metrics = evaluate_on_dataset(model, data_yaml, imgsz, device, log)
        results.update(det_metrics)
    else:
        log.warning(f"Dataset YAML not found: {data_yaml} — skipping detection metrics")

    # FPS benchmark
    if data_yaml.exists():
        speed_metrics = evaluate_fps(model, data_yaml, imgsz, device, log)
        results.update(speed_metrics)

    # False positive rate
    if args.clean_video:
        fp_metrics = evaluate_false_positive_rate(
            model, args.clean_video, cfg, imgsz, device, log
        )
        results.update(fp_metrics)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nResults saved → {output_path}")
    log.info("\n── Summary ──")
    for k, v in results.items():
        if k not in ("model", "model_path"):
            log.info(f"  {k:<25}: {v}")


if __name__ == "__main__":
    main()
