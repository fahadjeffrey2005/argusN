"""
sweep.py — SHARP parameter sweep.
====================================
Tests every combination of confirm_frames × conf threshold on two videos:
  - fod_video  : known FOD — higher dets/min = better recall
  - clean_video: clean runway — lower dets/min = better precision (fewer FPs)

The best config maximises (dets on fod - dets on clean).

Usage (from argusN/ root):
    python sharp/scripts/sweep.py \\
        --fod_video   data/raw/videos/fod1.mp4 \\
        --clean_video data/raw/videos/clean1.mp4

    # Custom ranges:
    python sharp/scripts/sweep.py \\
        --fod_video   data/raw/videos/fod1.mp4 \\
        --clean_video data/raw/videos/clean1.mp4 \\
        --confirm     2 3 4 \\
        --conf        0.20 0.25 0.30 0.35 0.40 \\
        --max_miss    2 3
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sharp.src.pipeline.sharp_pipeline import SharpPipeline, load_config


def run_config(base_cfg: dict, video_path: str) -> dict:
    """Run pipeline with a config dict and return stats."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(base_cfg, f)
        tmp = f.name
    try:
        pipe  = SharpPipeline(tmp)
        stats = pipe.run_video(video_path)
    finally:
        os.unlink(tmp)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fod_video",   required=True, help="Video with real FOD")
    parser.add_argument("--clean_video", required=True, help="Clean runway video (no FOD)")
    parser.add_argument("--config",      default="sharp/config/sharp_config.yaml")
    parser.add_argument("--confirm",     type=int,   nargs="+", default=[2, 3, 4])
    parser.add_argument("--conf",        type=float, nargs="+", default=[0.20, 0.25, 0.30, 0.35])
    parser.add_argument("--max_miss",    type=int,   nargs="+", default=[2, 3])
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    combos = [
        (cf, co, mm)
        for cf in args.confirm
        for co in args.conf
        for mm in args.max_miss
    ]

    print(f"\nSHARP parameter sweep — {len(combos)} combinations")
    print(f"  FOD video   : {args.fod_video}")
    print(f"  Clean video : {args.clean_video}")
    print(f"  confirm_frames : {args.confirm}")
    print(f"  conf           : {args.conf}")
    print(f"  max_miss       : {args.max_miss}")
    print()

    COL = 72
    hdr = f"{'confirm':>7}  {'conf':>5}  {'miss':>4}  | {'FOD det/min':>11}  {'Clean det/min':>13}  {'Signal':>8}  {'FP reduc':>8}"
    print(hdr)
    print("-" * len(hdr))

    results = []
    for i, (cf, co, mm) in enumerate(combos):
        cfg = copy.deepcopy(base_cfg)
        cfg["tracker"]["confirm_frames"] = cf
        cfg["model"]["conf"]             = co
        cfg["tracker"]["max_miss"]       = mm

        fod_s   = run_config(cfg, args.fod_video)
        clean_s = run_config(cfg, args.clean_video)

        fod_dpm   = fod_s["dets_per_min"]
        clean_dpm = clean_s["dets_per_min"]
        signal    = fod_dpm - clean_dpm       # higher = better
        fp_reduc  = fod_s["total_reduction"]

        row = {"confirm": cf, "conf": co, "max_miss": mm,
               "fod_dpm": fod_dpm, "clean_dpm": clean_dpm,
               "signal": signal, "fp_reduction": fp_reduc}
        results.append(row)

        marker = " ◀ BEST" if i == 0 else ""
        print(f"  {cf:>5}    {co:>5.2f}    {mm:>2}  | {fod_dpm:>11.1f}  {clean_dpm:>13.1f}  {signal:>8.1f}  {fp_reduc:>8}{marker}")

    print()
    best = max(results, key=lambda r: r["signal"])
    print("=" * 65)
    print("  BEST CONFIG (highest signal = FOD dets - clean dets)")
    print("=" * 65)
    print(f"  confirm_frames : {best['confirm']}")
    print(f"  conf threshold : {best['conf']}")
    print(f"  max_miss       : {best['max_miss']}")
    print(f"  FOD dets/min   : {best['fod_dpm']}")
    print(f"  Clean dets/min : {best['clean_dpm']}")
    print(f"  Signal         : {best['signal']:.1f}")
    print(f"  FP reduction   : {best['fp_reduction']}")
    print("=" * 65)
    print()
    print("To apply this config, edit sharp/config/sharp_config.yaml:")
    print(f"  tracker.confirm_frames: {best['confirm']}")
    print(f"  model.conf: {best['conf']}")
    print(f"  tracker.max_miss: {best['max_miss']}")
    print()


if __name__ == "__main__":
    main()
