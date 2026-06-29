#!/usr/bin/env python3
"""
enhanced/scripts/run_video.py
------------------------------
Run ENHANCED pipeline on a local video file.

Usage:
    python enhanced/scripts/run_video.py --video fod1.mp4
    python enhanced/scripts/run_video.py --video fod1.mp4 --out runs/fod1_enhanced.mp4
    python enhanced/scripts/run_video.py --video fod1.mp4 --show
    python enhanced/scripts/run_video.py --video clean1.mp4 --out runs/clean1_enhanced.mp4

Run from argusN/ root on Ubuntu:
    source venv/bin/activate
    python enhanced/scripts/run_video.py --video fod1.mp4 --out runs/fod1_v11s.mp4
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # argusN/
sys.path.insert(0, str(ROOT))

from enhanced.src.pipeline.enhanced_pipeline import EnhancedPipeline

CONFIG_DEFAULT = ROOT / "enhanced" / "config" / "enhanced_config.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="ENHANCED pipeline on a local video file")
    p.add_argument("--video",  required=True,  help="Path to input video (.mp4 etc.)")
    p.add_argument("--config", default=str(CONFIG_DEFAULT))
    p.add_argument("--out",    default=None,   help="Save annotated output video here (.mp4)")
    p.add_argument("--show",   action="store_true", help="Display live window (requires display)")
    p.add_argument("--device", default=None,   help="Override device: cpu / cuda")
    p.add_argument("--max-frames", type=int, default=None, help="Process only N frames (for quick test)")
    return p.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    config_path = args.config

    # Device override — patch config inline if needed
    if args.device:
        import yaml, tempfile, os
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["model"]["device"] = args.device
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(cfg, tmp)
        tmp.close()
        config_path = tmp.name

    print("")
    print("=== ENHANCED Pipeline — Video Mode ===")
    print(f"  Video  : {video_path}")
    print(f"  Config : {config_path}")
    print(f"  Output : {args.out or 'none (no video saved)'}")
    print(f"  Display: {'yes' if args.show else 'no'}")
    print("")

    pipeline = EnhancedPipeline(config_path)

    result = pipeline.run_video(
        video_path  = str(video_path),
        out_video   = args.out,
        show        = args.show,
        max_frames  = args.max_frames,
    )

    print("")
    print("=== Results ===")
    print(f"  Video            : {result['video']}")
    print(f"  Frames processed : {result['frames']}")
    print(f"  Duration         : {result['duration_s']}s")
    print(f"  Avg FPS          : {result['fps']}")
    print(f"  Avg latency      : {result['latency_ms']} ms/frame")
    print(f"  Total raw dets   : {result['total_raw']}")
    print(f"  Unique FOD found : {result['unique_fod_detected']}")
    if args.out:
        print(f"  Saved to         : {args.out}")
    print("")

    # Interpretation
    n = result['unique_fod_detected']
    if n == 0:
        print("  >> Zero FOD detected — check weights path and config.")
    elif n < 3:
        print(f"  >> {n} unique FOD object(s) confirmed by pipeline.")
    else:
        print(f"  >> {n} unique FOD objects — review output video for FP/FN.")


if __name__ == "__main__":
    main()
