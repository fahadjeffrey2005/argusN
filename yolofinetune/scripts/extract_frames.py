"""
extract_frames.py — Extract frames from raw runway video at a fixed FPS.

Extracts at 2fps by default to avoid near-identical frames.
Handles both clean tarmac video and FOD session recordings.

Usage:
    python scripts/extract_frames.py \
        --video data/raw/videos/fod_recording.mp4 \
        --output data/annotated/images/train \
        --fps 2

    # For clean tarmac baseline (no annotation needed):
    python scripts/extract_frames.py \
        --video data/raw/videos/clean_runway.mp4 \
        --output data/raw/images \
        --fps 2
"""

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: float = 2.0,
    prefix: str = "",
    start_frame: int = 0,
    max_frames: int = None
) -> int:
    """
    Extract frames from a video at target_fps.
    Returns the number of frames saved.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, round(source_fps / target_fps))

    print(f"Video   : {video_path.name}")
    print(f"Source  : {source_fps:.1f} fps, {total_frames} frames")
    print(f"Extract : every {frame_interval} frames → ~{source_fps / frame_interval:.1f} fps")
    print(f"Output  : {output_dir}")

    stem = prefix if prefix else video_path.stem
    saved = 0
    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pbar = tqdm(total=total_frames - start_frame, desc="Extracting", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = output_dir / f"{stem}_{frame_idx:06d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved += 1

            if max_frames and saved >= max_frames:
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"\nDone — {saved} frames saved to {output_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a runway video for annotation."
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory to save extracted frames"
    )
    parser.add_argument(
        "--fps", type=float, default=2.0,
        help="Extraction rate in frames per second (default: 2.0)"
    )
    parser.add_argument(
        "--prefix", default="",
        help="Filename prefix (default: video stem)"
    )
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Frame number to start extraction from (default: 0)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum number of frames to extract (default: all)"
    )
    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        target_fps=args.fps,
        prefix=args.prefix,
        start_frame=args.start_frame,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
