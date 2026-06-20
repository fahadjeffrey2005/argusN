"""
ARGUS-N — compare_all.py
Reads eval_results.json from all three models and prints a side-by-side
comparison table for the research paper.

Usage (from argusN root):
    python eval/compare_all.py
    python eval/compare_all.py --output eval/comparison_results.md
"""

import json
import argparse
from pathlib import Path

MODELS = [
    ("YOLOFINETUNE", "yolofinetune/logs/eval_results.json"),
    ("HAWKEYE",      "hawkeye/logs/eval_results.json"),
    ("PRIME",        "prime/logs/eval_results.json"),
]

METRICS = [
    ("mAP50",                    "mAP50",                    "{:.4f}"),
    ("mAP50-95",                 "mAP50_95",                 "{:.4f}"),
    ("Precision",                "precision",                "{:.4f}"),
    ("Recall",                   "recall",                   "{:.4f}"),
    ("F1",                       "f1",                       "{:.4f}"),
    ("False Positive Rate /min", "fp_per_minute",            "{:.3f}"),
    ("Avg Latency (ms)",         "latency_ms",               "{:.1f}"),
    ("Avg FPS",                  "fps",                      "{:.1f}"),
]


def load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def fmt(val, template):
    if val is None or val == "N/A" or val == "":
        return "—"
    try:
        return template.format(float(val))
    except (TypeError, ValueError):
        return str(val)


def build_table(results: dict) -> str:
    col_w = 26
    name_w = 30

    header = f"{'Metric':<{name_w}}" + "".join(f"{name:>{col_w}}" for name, _ in results.items())
    sep    = "-" * (name_w + col_w * len(results))
    rows   = [header, sep]

    for label, key, template in METRICS:
        row = f"{label:<{name_w}}"
        for model_name, data in results.items():
            val = data.get(key)
            row += f"{fmt(val, template):>{col_w}}"
        rows.append(row)

    return "\n".join(rows)


def build_markdown(results: dict) -> str:
    lines = []
    lines.append("# ARGUS-N — Model Comparison\n")

    # Header
    lines.append("| Metric | " + " | ".join(results.keys()) + " |")
    lines.append("|---|" + "|".join(["---"] * len(results)) + "|")

    for label, key, template in METRICS:
        row = f"| {label} |"
        for data in results.values():
            val = data.get(key)
            row += f" {fmt(val, template)} |"
        lines.append(row)

    lines.append("")
    lines.append("## Notes")
    lines.append("")

    # Auto-highlight winners
    for label, key, template in METRICS:
        vals = {}
        for name, data in results.items():
            v = data.get(key)
            if v is not None:
                try:
                    vals[name] = float(v)
                except (TypeError, ValueError):
                    pass
        if not vals:
            continue

        # Lower is better for FP rate and latency
        lower_is_better = key in ("fp_per_minute", "latency_ms")
        winner = min(vals, key=vals.__getitem__) if lower_is_better else max(vals, key=vals.__getitem__)
        lines.append(f"- **{label}**: best = **{winner}** ({fmt(vals[winner], template)})")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare all ARGUS-N model results")
    parser.add_argument("--output", default=None,
                        help="Write markdown table to this file (optional)")
    args = parser.parse_args()

    results = {}
    for name, path in MODELS:
        data = load(path)
        results[name] = data
        status = "✓" if data else "✗ not found"
        print(f"  {name:<16}: {path}  [{status}]")

    print()
    print(build_table(results))
    print()

    if args.output:
        md = build_markdown(results)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(md)
        print(f"Markdown saved → {args.output}")


if __name__ == "__main__":
    main()
