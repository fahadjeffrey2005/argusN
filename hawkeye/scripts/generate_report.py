"""
generate_report.py — Read eval_results.json and write hawkeye_results.md.

Loads HAWKEYE results and, if available, YOLOFINETUNE results for comparison.
Writes a Markdown report with a metrics table and commentary.

Usage (run from inside hawkeye/ directory):
    python scripts/generate_report.py

    # Custom paths:
    python scripts/generate_report.py \\
        --hawkeye-results logs/eval_results.json \\
        --yolo-results ../yolofinetune/logs/eval_results.json \\
        --output logs/hawkeye_results.md
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime


METRIC_LABELS = {
    "mAP50":                  "mAP50",
    "mAP50_95":               "mAP50-95",
    "precision":              "Precision",
    "recall":                 "Recall",
    "f1":                     "F1 Score",
    "fp_per_minute":          "False Positive Rate (per min)",
    "latency_ms":             "Latency (ms/frame)",
    "fps":                    "Inference FPS",
}


def load_results(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def fmt(val, key: str) -> str:
    """Format a metric value for the table."""
    if val is None:
        return "—"
    if key in ("mAP50", "mAP50_95", "precision", "recall", "f1"):
        return f"{float(val):.4f}"
    if key in ("fp_per_minute", "latency_ms"):
        return f"{float(val):.2f}"
    if key == "fps":
        return f"{float(val):.1f}"
    return str(val)


def delta_str(hawkeye_val, yolo_val, key: str, higher_is_better: bool) -> str:
    """Return a ▲/▼ delta string."""
    if hawkeye_val is None or yolo_val is None:
        return ""
    try:
        diff = float(hawkeye_val) - float(yolo_val)
    except (TypeError, ValueError):
        return ""
    if abs(diff) < 1e-6:
        return " (=)"
    arrow = "▲" if diff > 0 else "▼"
    good  = (diff > 0) == higher_is_better
    sign  = "+" if diff > 0 else ""
    if key in ("mAP50", "mAP50_95", "precision", "recall", "f1"):
        return f" ({arrow} {sign}{diff:.4f}{'✓' if good else '✗'})"
    if key in ("fp_per_minute", "latency_ms"):
        return f" ({arrow} {sign}{diff:.2f}{'✓' if good else '✗'})"
    if key == "fps":
        return f" ({arrow} {sign}{diff:.1f}{'✓' if good else '✗'})"
    return ""


# higher_is_better for each metric
HIGHER_IS_BETTER = {
    "mAP50":        True,
    "mAP50_95":     True,
    "precision":    True,
    "recall":       True,
    "f1":           True,
    "fp_per_minute": False,
    "latency_ms":   False,
    "fps":          True,
}


def generate_report(hawkeye: dict, yolo: dict, output: Path) -> str:
    has_yolo = bool(yolo)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append("# HAWKEYE — Evaluation Report")
    lines.append(f"\n_Generated: {now}_\n")

    lines.append("## Model")
    lines.append("**HAWKEYE** — Multi-stack FOD detector: fine-tuned YOLOv8 + Farneback optical flow "
                 "egomotion residual + PatchCore unsupervised anomaly detection. "
                 "Alert raised when 2 or more of 3 components vote positive.")
    lines.append("")

    # ── Metrics table ──────────────────────────────────────────────────────
    lines.append("## Results\n")

    if has_yolo:
        lines.append(f"| Metric | HAWKEYE | YOLOFINETUNE | Delta |")
        lines.append(f"|--------|---------|--------------|-------|")
        for key, label in METRIC_LABELS.items():
            hv = hawkeye.get(key)
            yv = yolo.get(key)
            d  = delta_str(hv, yv, key, HIGHER_IS_BETTER[key])
            lines.append(f"| {label} | {fmt(hv, key)} | {fmt(yv, key)} | {d} |")
    else:
        lines.append(f"| Metric | HAWKEYE |")
        lines.append(f"|--------|---------|")
        for key, label in METRIC_LABELS.items():
            hv = hawkeye.get(key)
            lines.append(f"| {label} | {fmt(hv, key)} |")

    lines.append("")

    # ── Commentary ─────────────────────────────────────────────────────────
    lines.append("## Commentary\n")

    # mAP comparison
    if has_yolo:
        h_map = hawkeye.get("mAP50")
        y_map = yolo.get("mAP50")
        if h_map is not None and y_map is not None:
            diff = float(h_map) - float(y_map)
            if diff > 0.01:
                lines.append(f"**Detection (mAP50):** HAWKEYE improves on YOLOFINETUNE by "
                              f"{diff:.4f} ({h_map:.4f} vs {y_map:.4f}). "
                              "The additional flow and PatchCore components boost recall on "
                              "borderline detections that YOLO misses alone.")
            elif diff < -0.01:
                lines.append(f"**Detection (mAP50):** HAWKEYE scores slightly below YOLOFINETUNE "
                              f"({h_map:.4f} vs {y_map:.4f}, Δ={diff:.4f}). "
                              "This is expected — the fusion gate can suppress correct YOLO detections "
                              "if the other two components don't agree.")
            else:
                lines.append(f"**Detection (mAP50):** HAWKEYE and YOLOFINETUNE are comparable "
                              f"({h_map:.4f} vs {y_map:.4f}).")
            lines.append("")

        # FP rate comparison
        h_fp = hawkeye.get("fp_per_minute")
        y_fp = yolo.get("fp_per_minute")
        if h_fp is not None and y_fp is not None:
            diff = float(h_fp) - float(y_fp)
            if diff < -0.1:
                lines.append(f"**False Positive Rate:** HAWKEYE reduces FP rate from "
                              f"{y_fp:.2f} to {h_fp:.2f} alerts/min "
                              f"(improvement of {abs(diff):.2f}/min). "
                              "The 2-of-3 voting gate suppresses single-component noise effectively.")
            elif diff > 0.1:
                lines.append(f"**False Positive Rate:** HAWKEYE has a higher FP rate than YOLOFINETUNE "
                              f"({h_fp:.2f} vs {y_fp:.2f} alerts/min). "
                              "Shadows and wet tarmac cause simultaneous false votes from flow + PatchCore — "
                              "a known weakness motivating PRIME's semantic classifier.")
            else:
                lines.append(f"**False Positive Rate:** Comparable to YOLOFINETUNE "
                              f"({h_fp:.2f} vs {y_fp:.2f} alerts/min).")
            lines.append("")

        # FPS comparison
        h_fps = hawkeye.get("fps")
        y_fps = yolo.get("fps")
        if h_fps is not None and y_fps is not None:
            lines.append(f"**Inference Speed:** HAWKEYE runs at {h_fps:.1f} fps vs "
                         f"YOLOFINETUNE's {y_fps:.1f} fps. "
                         "The PatchCore nearest-neighbour search accounts for the additional latency (~15-30ms). "
                         "PRIME addresses this with a fixed-size CNN classifier.")
            lines.append("")
    else:
        lines.append("_YOLOFINETUNE results not found — run without comparison._\n")

    # ── Known weaknesses ───────────────────────────────────────────────────
    lines.append("## Known Weaknesses (by design)\n")
    lines.append("These weaknesses are the motivation for PRIME:\n")
    lines.append("- **Sunny-day shadows** — flow + PatchCore both flag simultaneously → false alert passes 2/3 vote")
    lines.append("- **Wet tarmac** — PatchCore sees texture change as anomalous → elevated FP rate")
    lines.append("- **Strobe lights** — flow detects periodic brightness change → false vote")
    lines.append("- **PatchCore latency** — nearest-neighbour search adds ~15-30ms overhead, capping FPS")
    lines.append("")

    # ── Files ──────────────────────────────────────────────────────────────
    lines.append("## Output Files\n")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `logs/eval_results.json` | Machine-readable metrics (consumed by compare_all.py) |")
    lines.append("| `outputs/demo_hawkeye_fod1.mp4` | Annotated demo video |")
    lines.append("| `models/patchcore/bank.pt` | PatchCore memory bank |")
    lines.append("| `models/yolo/finetuned/best.pt` | YOLO weights (copied from yolofinetune) |")
    lines.append("")

    report = "\n".join(lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate HAWKEYE evaluation report.")
    parser.add_argument("--hawkeye-results", default="logs/eval_results.json")
    parser.add_argument("--yolo-results",    default="../yolofinetune/logs/eval_results.json")
    parser.add_argument("--output",          default="logs/hawkeye_results.md")
    args = parser.parse_args()

    hawkeye = load_results(Path(args.hawkeye_results))
    yolo    = load_results(Path(args.yolo_results))

    if not hawkeye:
        print(f"ERROR: HAWKEYE results not found at {args.hawkeye_results}")
        print("Run evaluate_hawkeye.py first.")
        sys.exit(1)

    if not yolo:
        print(f"Note: YOLOFINETUNE results not found at {args.yolo_results} — report will be single-model.")

    report = generate_report(hawkeye, yolo, Path(args.output))

    print(report)
    print(f"\nReport saved → {args.output}")


if __name__ == "__main__":
    main()
