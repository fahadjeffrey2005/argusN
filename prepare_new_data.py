"""
prepare_new_data.py — Extract frames from new videos and pre-annotate with current model.
==========================================================================================

Workflow
--------
1. Extracts frames at 2fps from every fod_*.mp4 in the input folder.
2. Runs current best.pt on every frame → saves YOLO .txt label files.
3. Outputs a staging folder you can open in Roboflow / LabelImg / CVAT to
   review and correct annotations before merging into the main dataset.

IMPORTANT — runway_corner_16sep.mp4 is treated as CLEAN VIDEO (no FOD),
useful for HAWKEYE FP evaluation. It is NOT added to the staging folder.

Usage
-----
    python prepare_new_data.py \
        --video_dir "data/raw/new data" \
        --model yolofinetune/runs/train/best.pt \
        --out data/staging_new

    # Then open data/staging_new/ in Roboflow or LabelImg to review/correct.
    # After review, run merge_staging.py to fold into the main dataset.

Output layout
-------------
    data/staging_new/
        images/        ← extracted frames (.jpg)
        labels/        ← YOLO pre-annotation .txt files (review these!)
        preview/       ← side-by-side preview images with drawn boxes
        summary.json   ← per-video stats (frames extracted, detections found)
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def extract_and_annotate(
    video_path: Path,
    images_dir: Path,
    labels_dir: Path,
    preview_dir: Path,
    model,
    fps: float = 2.0,
    conf: float = 0.20,
) -> dict:
    """Extract frames at fps, run model, write YOLO labels and preview images."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, round(source_fps / fps))
    stem = video_path.stem

    print(f"\n{'─'*60}")
    print(f"  {video_path.name}")
    print(f"  {source_fps:.0f}fps | {total_frames} frames | extracting every {interval}th → ~{source_fps/interval:.1f}fps")

    saved = 0
    detections = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            fname = f"{stem}_{frame_idx:06d}"
            img_path = images_dir / f"{fname}.jpg"
            lbl_path = labels_dir / f"{fname}.txt"

            cv2.imwrite(str(img_path), frame)

            # Run model
            results = model.predict(frame, conf=conf, verbose=False)

            h, w = frame.shape[:2]
            lines = []
            preview_frame = frame.copy()

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # YOLO normalised format: class cx cy w h
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    detections += 1

                    # Draw on preview
                    cv2.rectangle(preview_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{r.names[cls]} {conf_val:.2f}"
                    cv2.putText(preview_frame, label, (int(x1), int(y1) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            lbl_path.write_text("\n".join(lines))
            cv2.imwrite(str(preview_dir / f"{fname}_preview.jpg"), preview_frame)

            saved += 1

        frame_idx += 1

    cap.release()
    print(f"  → {saved} frames, {detections} detections (conf ≥ {conf})")

    return {
        "video": video_path.name,
        "frames_extracted": saved,
        "total_detections": detections,
        "avg_detections_per_frame": round(detections / max(saved, 1), 2),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract + pre-annotate new FOD video data")
    parser.add_argument("--video_dir", required=True,
                        help="Folder containing fod_*.mp4 and runway_corner_*.mp4")
    parser.add_argument("--model",     required=True,
                        help="Path to current best.pt weights")
    parser.add_argument("--out",       default="data/staging_new",
                        help="Output staging directory (default: data/staging_new)")
    parser.add_argument("--fps",       type=float, default=2.0,
                        help="Frame extraction rate (default: 2.0)")
    parser.add_argument("--conf",      type=float, default=0.20,
                        help="Pre-annotation confidence threshold (default: 0.20, intentionally low)")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    class_names = model.names
    print(f"Model loaded: {args.model}")
    print(f"Classes: {class_names}")

    video_dir = Path(args.video_dir)
    out = Path(args.out)
    images_dir  = out / "images"
    labels_dir  = out / "labels"
    preview_dir = out / "preview"

    for d in [images_dir, labels_dir, preview_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Write classes.txt so labeling tools know the class names
    (out / "classes.txt").write_text("\n".join(class_names.values()))

    # Separate FOD training videos from clean/utility videos
    fod_videos   = sorted(video_dir.glob("fod_*.mp4"))
    clean_videos = sorted(video_dir.glob("runway_corner_*.mp4"))

    if not fod_videos:
        print(f"No fod_*.mp4 files found in {video_dir}")
        return

    print(f"\nFound {len(fod_videos)} FOD training videos, {len(clean_videos)} clean/utility video(s)")
    if clean_videos:
        print(f"Clean videos (NOT added to staging — use for HAWKEYE evaluation):")
        for v in clean_videos:
            print(f"  {v.name}")

    summary = []
    for vp in fod_videos:
        stats = extract_and_annotate(vp, images_dir, labels_dir, preview_dir,
                                     model, fps=args.fps, conf=args.conf)
        summary.append(stats)

    # Write summary
    total_frames = sum(s["frames_extracted"] for s in summary)
    total_dets   = sum(s["total_detections"] for s in summary)
    summary_obj = {
        "model": args.model,
        "conf_threshold": args.conf,
        "fps_extracted": args.fps,
        "videos_processed": len(summary),
        "total_frames_extracted": total_frames,
        "total_detections": total_dets,
        "per_video": summary,
        "next_steps": [
            "1. Open staging_new/ in Roboflow, LabelImg, or CVAT",
            "2. Review preview/ images to see what the model found",
            "3. Correct labels: delete false positives, add missed FOD",
            "4. Run merge_staging.py to fold into main dataset",
            "5. Fine-tune: python yolofinetune/scripts/train.py --resume",
        ]
    }

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary_obj, indent=2))

    print(f"\n{'═'*60}")
    print(f"  DONE")
    print(f"  {total_frames} frames extracted from {len(summary)} videos")
    print(f"  {total_dets} pre-annotations written (conf ≥ {args.conf})")
    print(f"  Output: {out.resolve()}")
    print(f"{'═'*60}")
    print(f"\nNEXT: Review {out}/preview/ then correct labels in {out}/labels/")
    print(f"      Then run merge_staging.py to merge into main dataset.\n")


if __name__ == "__main__":
    main()
