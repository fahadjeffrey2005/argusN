"""
ARGUS-N Pre-Annotation Script

Runs first_finetuned_model.pt on all extracted images.
Generates YOLO format label files automatically.
You then import these into Roboflow to review and correct —
no drawing from scratch.

Usage:
    python scripts/preannotate.py

Output:
    data/raw/labels/  — one .txt file per image (YOLO format)
    data/raw/preannotation_report.txt — summary of what was found
"""

import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────
DRIVE_ROOT  = Path(__file__).resolve().parent.parent
MODEL_PATH  = DRIVE_ROOT / "raw data" / "first_finetuned_model.pt"
IMAGES_DIR  = DRIVE_ROOT / "data" / "raw" / "images"
LABELS_DIR  = DRIVE_ROOT / "data" / "raw" / "labels"
REPORT_PATH = DRIVE_ROOT / "data" / "raw" / "preannotation_report.txt"

LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Confidence threshold — lower = more boxes (more to review but fewer misses)
CONF = 0.25


def run():
    print("\n========================================")
    print("  ARGUS-N Pre-Annotation")
    print("========================================\n")

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        return

    images = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    if not images:
        print(f"[ERROR] No images found in: {IMAGES_DIR}")
        return

    print(f"Model  : {MODEL_PATH.name}")
    print(f"Images : {len(images)}")
    print(f"Conf   : {CONF}")
    print(f"Labels → {LABELS_DIR}\n")

    model = YOLO(str(MODEL_PATH))

    total_detections = 0
    frames_with_fod  = 0
    frames_clean     = 0

    for i, img_path in enumerate(images):
        results = model.predict(
            str(img_path),
            conf=CONF,
            verbose=False,
            device="mps"
        )

        label_path = LABELS_DIR / (img_path.stem + ".txt")
        result     = results[0]
        boxes      = result.boxes

        if boxes is None or len(boxes) == 0:
            # Clean frame — write empty label file
            label_path.write_text("")
            frames_clean += 1
        else:
            lines = []
            for box in boxes:
                cls_id = int(box.cls[0])
                xywhn  = box.xywhn[0].tolist()  # normalised cx cy w h
                cx, cy, w, h = xywhn
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            label_path.write_text("\n".join(lines))
            total_detections += len(boxes)
            frames_with_fod  += 1

        # Progress
        if (i + 1) % 100 == 0 or (i + 1) == len(images):
            print(f"  [{i+1}/{len(images)}] "
                  f"FOD frames so far: {frames_with_fod}")

    # ── Report ────────────────────────────────────────────
    report = (
        f"ARGUS-N Pre-Annotation Report\n"
        f"{'='*40}\n"
        f"Model             : {MODEL_PATH.name}\n"
        f"Total images      : {len(images)}\n"
        f"Frames with FOD   : {frames_with_fod}\n"
        f"Clean frames      : {frames_clean}\n"
        f"Total detections  : {total_detections}\n"
        f"Avg per FOD frame : "
        f"{total_detections/max(frames_with_fod,1):.1f}\n"
        f"{'='*40}\n"
    )
    REPORT_PATH.write_text(report)
    print(f"\n{report}")
    print(f"Labels saved to : {LABELS_DIR}")
    print(f"Report saved to : {REPORT_PATH}")
    print("\nNext steps:")
    print("  1. Go to https://roboflow.com")
    print("  2. Create new project → Object Detection")
    print("  3. Upload images + labels together (drag both folders)")
    print("  4. Review and correct boxes")
    print("  5. Export as YOLOv8 format")


if __name__ == "__main__":
    run()
