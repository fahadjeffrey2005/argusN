"""
ARGUS-N — FOD Frame Scanner

Plays through extracted images quickly.
Press F to flag a frame as containing FOD.
Press D to mark as definitely clean (skip).
At the end, copies flagged frames to data/annotate/
for focused annotation in Roboflow.

Controls:
    SPACE   — pause / resume auto-play
    F       — flag current frame (contains FOD)
    D       — clean frame (skip)
    A       — previous frame
    S       — next frame
    Q       — quit and save flagged list

Usage:
    python scripts/scan_frames.py
    python scripts/scan_frames.py --speed 5   # frames per second (default 8)
"""

import cv2
import shutil
import argparse
import time
from pathlib import Path

DRIVE_ROOT   = Path(__file__).resolve().parent.parent
IMAGES_DIR   = DRIVE_ROOT / "data" / "raw" / "images"
ANNOTATE_DIR = DRIVE_ROOT / "data" / "annotate"
FLAGGED_LOG  = DRIVE_ROOT / "data" / "raw" / "flagged_frames.txt"

ANNOTATE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=8,
                        help="Frames per second in auto-play (default: 8)")
    args = parser.parse_args()

    images = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    if not images:
        print(f"[ERROR] No images found in {IMAGES_DIR}")
        return

    total    = len(images)
    flagged  = []
    idx      = 0
    playing  = True
    delay    = 1.0 / args.speed

    cv2.namedWindow("ARGUS-N Scanner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ARGUS-N Scanner", 1280, 720)

    print(f"\nScanning {total} frames at {args.speed} fps")
    print("SPACE=pause  F=flag FOD  D=clean  A=prev  S=next  Q=quit\n")

    last_tick = time.time()

    while True:
        img_path = images[idx]
        frame    = cv2.imread(str(img_path))

        if frame is None:
            idx = min(idx + 1, total - 1)
            continue

        vis = frame.copy()
        is_flagged = img_path.name in [f.name for f in flagged]

        # Status overlay
        status_color = (0, 255, 0) if is_flagged else (200, 200, 200)
        status_text  = "FOD FLAGGED" if is_flagged else "clean"
        cv2.putText(vis,
                    f"[{idx+1}/{total}]  {img_path.stem}  |  {status_text}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    status_color, 2)
        cv2.putText(vis,
                    f"Flagged: {len(flagged)}  |  "
                    f"SPACE=pause  F=flag  D=clean  A/S=nav  Q=quit",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)

        if is_flagged:
            # Green border on flagged frames
            cv2.rectangle(vis, (0, 0),
                          (vis.shape[1]-1, vis.shape[0]-1),
                          (0, 255, 0), 6)

        cv2.imshow("ARGUS-N Scanner", vis)

        # Auto-advance timing
        now = time.time()
        elapsed = now - last_tick
        wait_ms = max(1, int((delay - elapsed) * 1000)) if playing else 0

        key = cv2.waitKey(wait_ms) & 0xFF
        last_tick = time.time()

        if key == ord('q'):
            break
        elif key == ord('f'):
            if img_path not in flagged:
                flagged.append(img_path)
                print(f"  Flagged: {img_path.name}  (total: {len(flagged)})")
            idx = min(idx + 1, total - 1)
        elif key == ord('d'):
            if img_path in flagged:
                flagged.remove(img_path)
            idx = min(idx + 1, total - 1)
        elif key == ord('a'):
            idx = max(idx - 1, 0)
        elif key == ord('s'):
            idx = min(idx + 1, total - 1)
        elif key == 32:  # SPACE
            playing = not playing
            print(f"  {'Playing' if playing else 'Paused'}")
        elif playing:
            idx = min(idx + 1, total - 1)
            if idx == total - 1:
                playing = False
                print("\n  End of frames — press Q to finish")

    cv2.destroyAllWindows()

    if not flagged:
        print("\nNo frames flagged.")
        return

    # ── Copy flagged frames to annotate folder ────────────
    print(f"\nCopying {len(flagged)} flagged frames to {ANNOTATE_DIR}...")
    for f in flagged:
        shutil.copy(f, ANNOTATE_DIR / f.name)

    # Save log
    FLAGGED_LOG.write_text("\n".join(f.name for f in flagged))

    print(f"\n{'='*45}")
    print(f"Flagged frames : {len(flagged)}")
    print(f"Saved to       : {ANNOTATE_DIR}")
    print(f"\nUpload ONLY the flagged folder to Roboflow:")
    print(f"  {ANNOTATE_DIR}")
    print(f"\nAnnotate just these frames — much faster.")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    main()
