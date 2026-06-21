"""
run_enhanced.py — Run ENHANCED on a video.
==========================================

Usage (from argusN/ root):
    python enhanced/scripts/run_enhanced.py --video data/raw/videos/fod1.mp4
    python enhanced/scripts/run_enhanced.py --video "data/raw/new data/fod_16sep00.mp4"
    python enhanced/scripts/run_enhanced.py --video data/raw/videos/clean1.mp4 --show_rejected
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from enhanced.src.pipeline.enhanced_pipeline import EnhancedPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",         required=True)
    parser.add_argument("--config",        default="enhanced/config/enhanced_config.yaml")
    parser.add_argument("--out_video",     default=None)
    parser.add_argument("--out_images",    default=None)
    parser.add_argument("--out_labels",    default=None)
    parser.add_argument("--show_rejected", action="store_true")
    parser.add_argument("--max_frames",    type=int, default=None)
    args = parser.parse_args()

    pipe       = EnhancedPipeline(args.config)
    video_path = Path(args.video)
    out_video  = args.out_video or f"enhanced_{video_path.stem}.mp4"

    print(f"\nENHANCED — {video_path.name}")
    print(f"Config  : {args.config}")
    print(f"Output  : {out_video}\n")

    stats = pipe.run_video(
        video_path    = str(video_path),
        out_video     = out_video,
        out_images    = args.out_images,
        out_labels    = args.out_labels,
        show_rejected = args.show_rejected,
        max_frames    = args.max_frames,
    )

    W = 55
    print("\n" + "=" * W)
    print(f"  ENHANCED RESULTS — {video_path.stem}")
    print("=" * W)
    print(f"  Video duration    : {stats['duration_s']}s")
    print(f"  Processing time   : {stats['processing_time_s']}s")
    print(f"  FPS               : {stats['fps']}")
    print(f"  Latency/frame     : {stats['latency_ms']} ms")
    print(f"  Raw YOLO dets     : {stats['total_raw']}")
    print(f"  ")
    print(f"  *** Unique FOD objects detected : {stats['unique_fod_detected']} ***")
    print(f"  (each physical object counted once, not per-frame)")
    print("=" * W)
    print(f"  Video saved → {out_video}")
    print("=" * W)


if __name__ == "__main__":
    main()
