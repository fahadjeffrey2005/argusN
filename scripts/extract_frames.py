"""
ARGUS-N — Frame Extractor for CVAT Annotation

Extracts frames from all new recordings using saved crop config.
Run this after crop_videos.py has set the crop line.

If no crop config exists, extracts full frames (no crop).

Usage:
    python scripts/extract_frames.py
    python scripts/extract_frames.py --every 10   # extract every 10th frame
    python scripts/extract_frames.py --every 5    # denser extraction
"""

import cv2
import argparse
from pathlib import Path

DRIVE_ROOT  = Path(__file__).resolve().parent.parent
RAW_DATA    = DRIVE_ROOT / "raw data"
OUT_IMAGES  = DRIVE_ROOT / "data" / "raw" / "images"
CROP_CONFIG = DRIVE_ROOT / "data" / "raw" / "crop_config.txt"

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
NEW_VIDEOS = sorted(RAW_DATA.glob("recording_20250521_*.mp4"))


def load_crop_config() -> dict:
    config = {}
    if CROP_CONFIG.exists():
        with open(CROP_CONFIG) as f:
            for line in f:
                line = line.strip()
                if "," in line:
                    name, y = line.split(",", 1)
                    config[name.strip()] = int(y.strip())
    return config


def extract(video_path: Path, crop_y: int, every_n: int):
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    saved = 0
    idx   = 0

    print(f"\n{video_path.name}")
    print(f"  Total frames : {total}")
    print(f"  FPS          : {fps:.1f}")
    print(f"  Crop y       : {crop_y if crop_y else 'none'}")
    print(f"  Sampling     : every {every_n} frames "
          f"(~{total // every_n} images)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            if crop_y:
                frame = frame[crop_y:, :]
            fname = f"{video_path.stem}_{idx:06d}.jpg"
            cv2.imwrite(str(OUT_IMAGES / fname), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        idx += 1

    cap.release()
    print(f"  Saved        : {saved} frames")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", type=int, default=15,
                        help="Extract every Nth frame (default: 15)")
    args = parser.parse_args()

    if not NEW_VIDEOS:
        print("[ERROR] No new recordings found.")
        return

    crop_config = load_crop_config()
    if crop_config:
        print(f"Loaded crop config for {len(crop_config)} video(s)")
    else:
        print("No crop config found — extracting full frames (no crop)")

    total = 0
    for video_path in NEW_VIDEOS:
        crop_y = crop_config.get(video_path.name, None)
        total += extract(video_path, crop_y, args.every)

    print(f"\n{'─'*45}")
    print(f"Total images extracted : {total}")
    print(f"Saved to               : {OUT_IMAGES}")
    print(f"\nUpload this folder to CVAT:")
    print(f"  1. Go to https://app.cvat.ai")
    print(f"  2. Create new task → upload images from:")
    print(f"     {OUT_IMAGES}")
    print(f"  3. Label class: 'fod'")
    print(f"  4. Export as: YOLO 1.1")
    print(f"{'─'*45}")


if __name__ == "__main__":
    main()
