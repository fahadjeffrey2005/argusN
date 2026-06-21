"""
run_sharp.py — Run SHARP on a video and get annotated output + stats.
======================================================================

Usage (from argusN/ root):
    python sharp/scripts/run_sharp.py --video data/raw/videos/fod1.mp4
    python sharp/scripts/run_sharp.py --video "data/raw/new data/fod_16sep00.mp4" --out_video sharp_16sep00.mp4
    python sharp/scripts/run_sharp.py --video data/raw/videos/clean1.mp4 --show_rejected
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sharp.src.pipeline.sharp_pipeline import SharpPipeline


def main():
    parser = argparse.ArgumentParser(description="Run SHARP on a video")
    parser.add_argument("--video",        required=True)
    parser.add_argument("--config",       default="sharp/config/sharp_config.yaml")
    parser.add_argument("--out_video",    default=None,
                        help="Output video path (default: sharp_<stem>.mp4)")
    parser.add_argument("--out_images",   default=None)
    parser.add_argument("--out_labels",   default=None)
    parser.add_argument("--show_rejected", action="store_true",
                        help="Draw rejected detections in grey (debug mode)")
    args = parser.parse_args()

    pipe = SharpPipeline(args.config)

    video_path = Path(args.video)
    out_video  = args.out_video or f"sharp_{video_path.stem}.mp4"

    print(f"\nSHARP — {video_path.name}")
    print(f"Config  : {args.config}")
    print(f"Output  : {out_video}")
    if args.show_rejected:
        print("Mode    : DEBUG (rejected detections shown in grey)")
    print()

    stats = pipe.run_video(
        video_path   = str(video_path),
        out_video    = out_video,
        out_images   = args.out_images,
        out_labels   = args.out_labels,
        show_rejected= args.show_rejected,
    )

    W = 55
    print("=" * W)
    print(f"  SHARP RESULTS — {video_path.stem}")
    print("=" * W)
    print(f"  Video duration    : {stats['duration_s']}s")
    print(f"  Processing time   : {stats['processing_time_s']}s")
    print(f"  FPS               : {stats['fps']}")
    print(f"  Latency/frame     : {stats['latency_ms']} ms")
    print(f"  Raw YOLO dets     : {stats['total_raw']}")
    print(f"  After tracker     : {stats['total_confirmed']}  ({stats['tracker_reduction']} suppressed)")
    print(f"  After post-filter : {stats['total_passed']}  ({stats['filter_reduction']} suppressed)")
    print(f"  Total FP reduction: {stats['total_reduction']}")
    print(f"  Detections/min    : {stats['dets_per_min']}")
    print(f"  Frames with dets  : {stats['frames_with_dets']}")
    print("=" * W)
    print(f"  Video saved → {out_video}")
    print("=" * W)


if __name__ == "__main__":
    main()
