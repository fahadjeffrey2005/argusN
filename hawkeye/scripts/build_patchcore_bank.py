"""
HAWKEYE — Build PatchCore Memory Bank

Extracts clean tarmac frames from a video (or loads from a directory),
runs them through the PatchCore backbone, and saves the feature bank.

Usage:
    # Extract frames from video and build bank in one pass
    python scripts/build_patchcore_bank.py \
        --video data/raw/videos/clean_runway.mp4 \
        --frames 100 \
        --output data/clean_frames \
        --save models/patchcore/bank.pt

    # Build bank from already-extracted frames
    python scripts/build_patchcore_bank.py \
        --images data/clean_frames \
        --save models/patchcore/bank.pt

Notes:
    - 60-100 clean frames is sufficient for a stable bank
    - Extract at even intervals across the full clean video
    - No labels needed — unsupervised
    - One-time build, takes ~2-3 minutes on CPU
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.anomaly.patchcore import PatchCore


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    num_frames: int = 100,
    logger=None
) -> list:
    """
    Extract evenly-spaced frames from a video file.
    Saves to output_dir and returns list of BGR frames.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)

    if logger:
        logger.info(
            f"Video: {total_frames} frames total, "
            f"extracting {num_frames} at step={step}"
        )

    frames = []
    frame_idx = 0
    saved = 0

    while saved < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        save_path = output_path / f"clean_{saved:04d}.jpg"
        cv2.imwrite(str(save_path), frame)
        frames.append(frame)
        saved += 1
        frame_idx += step

    cap.release()

    if logger:
        logger.info(f"Extracted {len(frames)} frames to {output_dir}")

    return frames


def load_frames_from_dir(image_dir: str, logger=None) -> list:
    """Load all images from a directory as BGR frames."""
    image_path = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([
        f for f in image_path.iterdir()
        if f.suffix.lower() in extensions
    ])

    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    frames = []
    for f in image_files:
        frame = cv2.imread(str(f))
        if frame is not None:
            frames.append(frame)

    if logger:
        logger.info(f"Loaded {len(frames)} frames from {image_dir}")

    return frames


def main():
    parser = argparse.ArgumentParser(description="Build PatchCore memory bank")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to clean tarmac video file")
    parser.add_argument("--images", type=str, default=None,
                        help="Path to directory of pre-extracted clean frames")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of frames to extract from video (default: 100)")
    parser.add_argument("--output", type=str, default="data/clean_frames",
                        help="Directory to save extracted frames (video mode)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save bank .pt file (overrides config)")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger(
        "build_patchcore_bank",
        cfg.get("logging", "log_path", default="logs/hawkeye.log"),
        cfg.get("logging", "level", default="INFO")
    )

    logger.info("=" * 50)
    logger.info("HAWKEYE — Building PatchCore Bank")
    logger.info("=" * 50)

    # Load or extract frames
    if args.video:
        if not Path(args.video).exists():
            logger.error(f"Video not found: {args.video}")
            sys.exit(1)
        frames = extract_frames_from_video(
            args.video, args.output, args.frames, logger
        )
    elif args.images:
        frames = load_frames_from_dir(args.images, logger)
    else:
        logger.error("Provide --video or --images")
        sys.exit(1)

    logger.info(f"Total frames for bank: {len(frames)}")

    # Build PatchCore and bank
    logger.info("Initialising PatchCore...")
    pc = PatchCore(cfg)

    logger.info("Building memory bank...")
    pc.build_bank(frames)

    # Save bank
    save_path = args.save or cfg.get("patchcore", "bank_path")
    pc.save_bank(save_path)

    logger.info("=" * 50)
    logger.info(f"Bank saved to: {save_path}")
    logger.info(f"Bank size: {pc.memory_bank.shape[0]} feature vectors")
    logger.info(f"Feature dim: {pc.memory_bank.shape[1]}")
    logger.info("=" * 50)
    logger.info("Done. Run scripts/run_hawkeye.py to start inference.")


if __name__ == "__main__":
    main()
