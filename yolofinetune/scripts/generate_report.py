"""
generate_report.py — Read eval_results.json and write yolofinetune_results.md.

Run after evaluate.py has completed.

Usage:
    python scripts/generate_report.py
"""

import json
from pathlib import Path
from datetime import datetime


EVAL_JSON   = Path("logs/eval_results.json")
REPORT_PATH = Path("yolofinetune_results.md")


def fmt(val, decimals=4, suffix=""):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}{suffix}"


def main():
    if not EVAL_JSON.exists():
        print(f"ERROR: {EVAL_JSON} not found. Run evaluate.py first.")
        return

    with open(EVAL_JSON) as f:
        r = json.load(f)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"""# YOLOFINETUNE — Baseline Results
### ARGUS-N Model 1 | Generated {now}

---

## Model Summary

| Field | Value |
|---|---|
| Model | YOLOFINETUNE — YOLOv8n fine-tuned |
| Architecture | YOLOv8n, backbone frozen (10 layers) |
| Classes | 1 — `fod` |
| Input size | 640 × 640 |
| Confidence threshold | 0.35 |
| IoU threshold | 0.45 |
| Epochs | 50 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Training device | CUDA (Ubuntu NVIDIA GPU) |

---

## Dataset

| Split | Images | Labels |
|---|---|---|
| Train (augmented ×5) | ~360 | ~360 |
| Validation | 15 | 15 |
| Test | 16 | 16 |
| Total annotated | 103 | 103 |

Source: Roboflow — `durvas-workspace-ihhkq/hawkeye-ap3a8 v1`
Original classes remapped → single class `0 = fod`

---

## Detection Metrics (Test Set)

| Metric | Value |
|---|---|
| **mAP50** | **{fmt(r.get('mAP50'), 4)}** |
| **mAP50-95** | **{fmt(r.get('mAP50_95'), 4)}** |
| Precision | {fmt(r.get('precision'), 4)} |
| Recall | {fmt(r.get('recall'), 4)} |
| F1 Score | {fmt(r.get('f1'), 4)} |

---

## Speed

| Metric | Value |
|---|---|
| Inference latency | {fmt(r.get('latency_ms'), 1, ' ms/frame')} |
| Inference FPS | {fmt(r.get('fps'), 1, ' fps')} |

---

## False Positive Rate (Clean Runway)

| Metric | Value |
|---|---|
| Clean video duration | {fmt(r.get('clean_video_duration_s'), 1, ' s')} |
| Total false alerts | {r.get('fp_total', '—')} |
| **False positive rate** | **{fmt(r.get('fp_per_minute'), 2, ' alerts/min')}** |

> The false positive rate on clean tarmac is the primary operational metric.
> A false alert stops the vehicle and wastes inspection time.

---

## Role in Comparative Study

YOLOFINETUNE is **Model 1 — the baseline**.
All metrics above are the performance floor.
HAWKEYE (Model 2) and PRIME (Model 3) are evaluated on the identical test set
and their results are compared directly against these numbers.

| Metric | YOLOFINETUNE | HAWKEYE | PRIME |
|---|---|---|---|
| mAP50 | {fmt(r.get('mAP50'), 4)} | — | — |
| mAP50-95 | {fmt(r.get('mAP50_95'), 4)} | — | — |
| Precision | {fmt(r.get('precision'), 4)} | — | — |
| Recall | {fmt(r.get('recall'), 4)} | — | — |
| F1 | {fmt(r.get('f1'), 4)} | — | — |
| FP rate (alerts/min) | {fmt(r.get('fp_per_minute'), 2)} | — | — |
| FPS | {fmt(r.get('fps'), 1)} | — | — |
| Latency (ms) | {fmt(r.get('latency_ms'), 1)} | — | — |

---

## Demo Video

`outputs/demo_yolofinetune_fod1.mp4`
Produced by running the trained model on `fod1.mp4`.
Used for side-by-side visual comparison with HAWKEYE and PRIME outputs.

---

## Weights

`models/yolo/finetuned/best.pt`
HAWKEYE and PRIME copy this file directly — no retraining.
"""

    REPORT_PATH.write_text(md)
    print(f"Report written → {REPORT_PATH}")


if __name__ == "__main__":
    main()
