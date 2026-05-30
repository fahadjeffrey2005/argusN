"""
ARGUS-N — Interactive Horizon Crop Tool

Opens the first frame of each new recording.
You click the horizon line (where road ends and sky/buildings begin).
The script saves the crop y-coordinate and applies it to all videos,
saving cropped frames to data/raw/images/ ready for CVAT annotation.

Usage:
    python scripts/crop_videos.py

Controls:
    - Click anywhere on the frame to set the crop line
    - Press ENTER to confirm and move to next video
    - Press R to reset the line on current video
    - Press S to skip current video
    - Press Q to quit
"""

import cv2
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
DRIVE_ROOT  = Path(__file__).resolve().parent.parent
RAW_DATA    = DRIVE_ROOT / "raw data"
OUT_IMAGES  = DRIVE_ROOT / "data" / "raw" / "images"
CROP_CONFIG = DRIVE_ROOT / "data" / "raw" / "crop_config.txt"

OUT_IMAGES.mkdir(parents=True, exist_ok=True)

# New recordings only (21 May 2025)
NEW_VIDEOS = sorted(RAW_DATA.glob("recording_20250521_*.mp4"))


# ── Interactive state ──────────────────────────────────────
crop_y = None
frame_display = None

def on_mouse(event, x, y, flags, param):
    global crop_y, frame_display
    if event == cv2.EVENT_LBUTTONDOWN:
        crop_y = y
        _redraw()

def _redraw():
    global frame_display
    if frame_display is None or crop_y is None:
        return
    vis = frame_display.copy()
    h, w = vis.shape[:2]
    # Draw crop line
    cv2.line(vis, (0, crop_y), (w, crop_y), (0, 255, 0), 2)
    cv2.putText(vis, f"Crop at y={crop_y}  |  ENTER=confirm  R=reset  S=skip",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, "Keep BELOW the green line",
                (10, crop_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("ARGUS-N Crop Tool", vis)


def get_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def extract_cropped_frames(
    video_path: Path,
    crop_y: int,
    every_n: int = 15,
    prefix: str = ""
):
    """
    Extract every Nth frame from video, crop from crop_y downward,
    save to OUT_IMAGES.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    idx   = 0

    print(f"  Extracting from {video_path.name} "
          f"({total} frames, every {every_n}th)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            cropped = frame[crop_y:, :]
            fname   = f"{prefix}_{idx:06d}.jpg"
            cv2.imwrite(str(OUT_IMAGES / fname), cropped)
            saved += 1
        idx += 1

    cap.release()
    print(f"  Saved {saved} frames → {OUT_IMAGES}")
    return saved


def main():
    global crop_y, frame_display

    if not NEW_VIDEOS:
        print("[ERROR] No recordings found in:", RAW_DATA)
        return

    print(f"\nFound {len(NEW_VIDEOS)} new recordings.")
    print("Click on each frame to set the horizon crop line.\n")

    cv2.namedWindow("ARGUS-N Crop Tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ARGUS-N Crop Tool", 1280, 720)
    cv2.setMouseCallback("ARGUS-N Crop Tool", on_mouse)

    crop_configs = {}
    total_saved  = 0

    for video_path in NEW_VIDEOS:
        print(f"\nVideo: {video_path.name}")
        frame = get_first_frame(video_path)
        if frame is None:
            print("  [SKIP] Could not read first frame")
            continue

        h = frame.shape[0]
        crop_y = h // 3  # default: top third removed
        frame_display = frame.copy()
        _redraw()

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 13:  # ENTER — confirm
                print(f"  Crop confirmed at y={crop_y}")
                crop_configs[video_path.name] = crop_y
                prefix = video_path.stem
                saved = extract_cropped_frames(video_path, crop_y, prefix=prefix)
                total_saved += saved
                break
            elif key == ord('r'):  # reset
                crop_y = h // 3
                _redraw()
            elif key == ord('s'):  # skip
                print("  Skipped")
                break
            elif key == ord('q'):  # quit
                print("\nQuitting early.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

    # Save crop config
    with open(CROP_CONFIG, "w") as f:
        for name, y in crop_configs.items():
            f.write(f"{name},{y}\n")
    print(f"\nCrop config saved to {CROP_CONFIG}")
    print(f"\nTotal frames extracted: {total_saved}")
    print(f"Images ready for CVAT: {OUT_IMAGES}")
    print("\nNext step:")
    print("  Upload the images folder to CVAT for annotation.")
    print("  https://app.cvat.ai")


if __name__ == "__main__":
    main()
