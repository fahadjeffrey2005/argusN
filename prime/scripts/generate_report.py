"""
PRIME — generate_report.py
Reads logs/eval_results.json and logs/train_history.json,
writes a formatted Markdown report to prime_results.md.

Usage (from inside prime/):
    python scripts/generate_report.py
    python scripts/generate_report.py --output prime_results.md
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


TEMPLATE = """\
# PRIME — Evaluation Results

**Model:** PRIME (YOLO + Farneback Flow + MobileNetV3-Small 4-channel CNN)
**Date:** {date}
**Video:** `{video}`

---

## Detection Metrics

| Metric | Value |
|---|---|
| Precision | {precision} |
| Recall | {recall} |
| F1 Score | {f1} |
| False Positive Rate (per min) | {fp_rate} |
| Avg Inference Latency (ms) | {latency_ms} |
| Avg FPS | {fps} |

---

## Raw Counts

| | Count |
|---|---|
| True Positives | {tp} |
| False Positives | {fp} |
| False Negatives | {fn} |
| Total Frames | {total_frames} |

---

## CNN Training History

{train_table}

Best validation loss: **{best_val_loss}** at epoch **{best_epoch}**

---

## Architecture Summary

```
Frame (T) + Frame (T-1)
    │                   │
    ▼                   ▼
YOLOv8n (fine-tuned)   Farneback Optical Flow
Full-frame detection   Egomotion subtraction
    │                   │
    └─────── Fusion ────┘
          Source tagging
     (both / yolo_only / flow_only)
                │
                ▼
    MobileNetV3-Small (4-channel)
    Channel 1-3: BGR patch
    Channel 4:   Flow magnitude
                │
    ┌───────────┴──────────────┐
    │ Classes: fod / shadow /  │
    │ runway_marking /         │
    │ strobe_light /           │
    │ clean_tarmac             │
    └──────────────────────────┘
    Only class 0 (fod) → ALERT
```

### Key design decisions
- YOLO and flow operate independently; CNN resolves semantic ambiguity
- Source tag `both` gets a confidence bonus — visual + physics agreement
- BumpDetector discards flow pathway when vehicle hits surface irregularity
- MobileNetV3-Small: ~3MB model, ~50fps on GPU, ~15fps on CPU

---

## Notes

{notes}
"""


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_train_table(history: list) -> tuple:
    if not history:
        return "_No training history found._", "N/A", "N/A"

    rows = ["| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |",
            "|---|---|---|---|---|"]
    best_loss = float("inf")
    best_epoch = 0
    for e in history:
        rows.append(
            f"| {e['epoch']} | {e['train_loss']:.4f} | {e['train_acc']:.3f} "
            f"| {e['val_loss']:.4f} | {e['val_acc']:.3f} |"
        )
        if e["val_loss"] < best_loss:
            best_loss  = e["val_loss"]
            best_epoch = e["epoch"]

    return "\n".join(rows), f"{best_loss:.4f}", str(best_epoch)


def main():
    parser = argparse.ArgumentParser(description="Generate PRIME results report")
    parser.add_argument("--eval-results", default="logs/eval_results.json")
    parser.add_argument("--train-history", default="logs/train_history.json")
    parser.add_argument("--output", default="prime_results.md")
    args = parser.parse_args()

    eval_data  = load_json(Path(args.eval_results))
    train_data = load_json(Path(args.train_history))

    if eval_data is None:
        print(f"ERROR: {args.eval_results} not found. Run evaluate_prime.py first.")
        sys.exit(1)

    train_table, best_val, best_epoch = build_train_table(train_data or [])

    notes_lines = []
    if eval_data.get("avg_fps", 0) > 40:
        notes_lines.append("- Inference FPS exceeds 40fps — suitable for real-time deployment.")
    if eval_data.get("false_positive_rate_per_min", 999) < 1.0:
        notes_lines.append("- False positive rate < 1/min — meets operational threshold.")
    if eval_data.get("f1", 0) > 0.9:
        notes_lines.append("- F1 > 0.90 — strong detection performance.")
    if not notes_lines:
        notes_lines = ["_Review metrics above and compare against YOLOFINETUNE and HAWKEYE baselines._"]

    report = TEMPLATE.format(
        date        = datetime.now().strftime("%Y-%m-%d %H:%M"),
        video       = eval_data.get("video", "N/A"),
        precision   = eval_data.get("precision",   "N/A"),
        recall      = eval_data.get("recall",      "N/A"),
        f1          = eval_data.get("f1",          "N/A"),
        fp_rate     = eval_data.get("false_positive_rate_per_min", "N/A"),
        latency_ms  = eval_data.get("avg_latency_ms", "N/A"),
        fps         = eval_data.get("avg_fps",     "N/A"),
        tp          = eval_data.get("total_tp",    "N/A"),
        fp          = eval_data.get("total_fp",    "N/A"),
        fn          = eval_data.get("total_fn",    "N/A"),
        total_frames = eval_data.get("total_frames", "N/A"),
        train_table = train_table,
        best_val_loss = best_val,
        best_epoch  = best_epoch,
        notes       = "\n".join(notes_lines),
    )

    Path(args.output).write_text(report)
    print(f"Report written → {args.output}")


if __name__ == "__main__":
    main()
