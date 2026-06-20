"""
evaluate_fod.py — Standardized FOD Model Evaluation Script
===========================================================
Computes a consistent set of metrics so different models can be fairly compared.

METRIC DEFINITIONS
------------------
mAP50          : mean Average Precision at IoU ≥ 0.5
                 Measures overall detector quality across all confidence thresholds.
                 A score of 0.83 means the model achieves 83% of a perfect
                 precision-recall curve when a detection overlaps ground truth by ≥ 50%.

mAP50-95       : mean Average Precision averaged over IoU thresholds 0.50–0.95 (step 0.05).
                 Stricter than mAP50 — penalises loose bounding boxes.

Precision      : Of all detections the model made, what fraction were actually FOD?
                 High precision = few false positives (per-image basis).

Recall         : Of all actual FOD objects in the test set, what fraction were detected?
                 THIS is what we mean by "71.4% efficient" — 71.4% recall means the model
                 found 71.4% of real FOD objects at IoU ≥ 0.5.

F1             : Harmonic mean of Precision and Recall. 2*P*R / (P+R).
                 Balances both — useful single-number summary.

Latency (ms)   : Average inference time per frame in milliseconds (GPU or CPU).
FPS            : Frames per second = 1000 / latency_ms.

FP/min         : False positives per minute on clean video (no FOD present).
                 Measures nuisance-alarm rate. Only computed if --clean is supplied.

USAGE
-----
Minimum (detection metrics only):
    python evaluate_fod.py --model best.pt --data dataset.yaml

With FP rate on clean video:
    python evaluate_fod.py --model best.pt --data dataset.yaml --clean clean_video.mp4

With custom output path:
    python evaluate_fod.py --model best.pt --data dataset.yaml --out my_results.json

REQUIREMENTS
------------
    pip install ultralytics opencv-python-headless
"""

import argparse
import json
import time
from pathlib import Path


# ─────────────────────────────────────────────
# 1. Detection metrics (mAP50, precision, recall, F1, latency, FPS)
# ─────────────────────────────────────────────

def run_detection_eval(model_path: str, data_yaml: str) -> dict:
    """Run YOLO val() and extract standardized metrics."""
    from ultralytics import YOLO

    print(f"\n[1/2] Running detection evaluation on dataset: {data_yaml}")
    model = YOLO(model_path)

    results = model.val(data=data_yaml, verbose=False)

    # --- mAP and detection scores ---
    map50     = float(results.box.map50)
    map50_95  = float(results.box.map)
    precision = float(results.box.mp)   # mean precision across classes
    recall    = float(results.box.mr)   # mean recall across classes
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # --- Latency ---
    # results.speed is a dict: {'preprocess': ms, 'inference': ms, 'postprocess': ms}
    latency_ms = float(results.speed.get("inference", 0.0))
    fps        = round(1000.0 / latency_ms, 2) if latency_ms > 0 else 0.0

    return {
        "mAP50":       round(map50,     4),
        "mAP50_95":    round(map50_95,  4),
        "precision":   round(precision, 4),
        "recall":      round(recall,    4),
        "f1":          round(f1,        4),
        "latency_ms":  round(latency_ms, 3),
        "fps":         fps,
    }


# ─────────────────────────────────────────────
# 2. FP rate on clean video
# ─────────────────────────────────────────────

def run_fp_eval(model_path: str, clean_video: str, conf: float = 0.25) -> dict:
    """
    Run the model on a clean video (no FOD present).
    Every detection is a false positive by definition.

    Args:
        conf: confidence threshold for counting a detection. Match whatever
              threshold you use at inference time (default 0.25).
    """
    import cv2
    from ultralytics import YOLO

    print(f"\n[2/2] Measuring FP rate on clean video: {clean_video}")
    model   = YOLO(model_path)
    cap     = cv2.VideoCapture(clean_video)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {clean_video}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / video_fps

    fp_total = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        preds = model.predict(frame, conf=conf, verbose=False)
        for r in preds:
            fp_total += len(r.boxes)
        frame_idx += 1
        if frame_idx % 300 == 0:
            pct = 100 * frame_idx / total_frames
            print(f"  {pct:.0f}% ({frame_idx}/{total_frames} frames) — FP so far: {fp_total}")

    cap.release()

    fp_per_minute = round(fp_total / (duration_s / 60.0), 4) if duration_s > 0 else 0.0

    return {
        "fp_total":               fp_total,
        "fp_per_minute":          fp_per_minute,
        "clean_video_duration_s": round(duration_s, 2),
    }


# ─────────────────────────────────────────────
# 3. Pretty-print results
# ─────────────────────────────────────────────

def print_table(metrics: dict, model_name: str):
    W = 52
    sep = "─" * W

    print(f"\n{'═' * W}")
    print(f"  FOD Evaluation Results — {model_name}")
    print(f"{'═' * W}")

    rows = [
        ("mAP50",         metrics.get("mAP50"),         "higher = better overall detector"),
        ("mAP50-95",      metrics.get("mAP50_95"),       "stricter box localisation quality"),
        ("Precision",     metrics.get("precision"),      "% of alerts that were real FOD"),
        ("Recall",        metrics.get("recall"),         "% of real FOD that was found"),
        ("F1 Score",      metrics.get("f1"),             "harmonic mean of P & R"),
        ("Latency (ms)",  metrics.get("latency_ms"),     "inference time per frame"),
        ("FPS",           metrics.get("fps"),            "frames processed per second"),
    ]

    if "fp_per_minute" in metrics:
        rows += [
            ("FP / minute",   metrics.get("fp_per_minute"),  "false alarms per min on clean video"),
            ("FP total",      metrics.get("fp_total"),        f"over {metrics.get('clean_video_duration_s', 0):.0f}s"),
        ]

    for label, value, note in rows:
        if value is None:
            continue
        val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {label:<18} {val_str:<10}  # {note}")

    print(sep)
    print(f"  Recall interpretation: the model detected")
    recall = metrics.get("recall", 0)
    print(f"  {recall*100:.1f}% of actual FOD objects at IoU ≥ 0.5")
    print(f"{'═' * W}\n")


# ─────────────────────────────────────────────
# 4. CLI entry point
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Standardized FOD model evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",  required=True,  help="Path to YOLO weights, e.g. best.pt")
    p.add_argument("--data",   required=True,  help="Path to dataset.yaml")
    p.add_argument("--clean",  default=None,   help="(Optional) Clean video path for FP rate")
    p.add_argument("--conf",   type=float, default=0.25,
                   help="Confidence threshold for FP counting (default 0.25)")
    p.add_argument("--out",    default=None,   help="Output JSON path (default: fod_eval_<model>.json)")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_name = model_path.stem

    # Detection metrics
    metrics = run_detection_eval(str(model_path), args.data)
    metrics["model"] = model_name

    # FP rate (optional)
    if args.clean:
        fp_metrics = run_fp_eval(str(model_path), args.clean, conf=args.conf)
        metrics.update(fp_metrics)

    # Print table
    print_table(metrics, model_name)

    # Save JSON
    out_path = Path(args.out) if args.out else Path(f"fod_eval_{model_name}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
