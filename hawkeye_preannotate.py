"""
hawkeye_preannotate.py — Run HAWKEYE pipeline on new videos for pre-annotation.
=================================================================================

Unlike prepare_new_data.py (which ran raw YOLO on still frames), this script
runs the full HAWKEYE pipeline — YOLO + temporal tracker — on each video.

WHY THIS IS BETTER FOR ANNOTATION:
    Raw YOLO sees each frame independently and flags shadows, cracks, and
    runway markings as FOD. HAWKEYE's temporal tracker requires the SAME box
    to appear in >= confirm_frames consecutive frames before it's confirmed.
    Real FOD is stationary — the camera approaches and YOLO sees it many frames
    in a row. Transient noise never reaches confirmation.

    Result: the pre-annotations in staging_hawkeye/ have far fewer false
    positives than staging_new/ — less cleanup work for the annotation team.

COORDINATE NOTE:
    HAWKEYE crops each frame (top 50%, bottom 5%) before running YOLO.
    This script converts detected boxes back to full-frame coordinates
    so the saved YOLO labels match the saved full-frame images.

Usage (run from /Volumes/T72/argusN):
    source hawkeye/venv/bin/activate  # or yolofinetune/venv
    python hawkeye_preannotate.py \
        --video_dir "data/raw/new data" \
        --model yolofinetune/models/yolo/finetuned/best.pt \
        --out data/staging_hawkeye

Output layout (same structure as staging_new for easy comparison):
    data/staging_hawkeye/
        images/         full-frame .jpg files at 2fps
        labels/         YOLO .txt — only HAWKEYE-confirmed detections
        preview/        images with confirmed boxes drawn
        summary.json    per-video stats
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ── Minimal inline tracker (no hawkeye/ import needed) ────────────────────────
# Copied from hawkeye/src/tracking/temporal_tracker.py so this script is fully
# standalone — works whether or not hawkeye/ venv is active.

class _Track:
    _id = 0
    def __init__(self, det, confirm_frames):
        _Track._id += 1
        self.tid   = _Track._id
        self.box   = det
        self.hits  = 1
        self.miss  = 0
        self.confirmed = False
        self.cf    = confirm_frames

    def update(self, det):
        self.box  = det
        self.hits += 1
        self.miss  = 0
        if self.hits >= self.cf:
            self.confirmed = True

    def mark_missed(self):
        self.miss += 1

    @property
    def x1(self): return self.box["x1"]
    @property
    def y1(self): return self.box["y1"]
    @property
    def x2(self): return self.box["x2"]
    @property
    def y2(self): return self.box["y2"]


def _iou(t, d):
    ix1 = max(t.x1, d["x1"]); iy1 = max(t.y1, d["y1"])
    ix2 = min(t.x2, d["x2"]); iy2 = min(t.y2, d["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (t.x2 - t.x1) * (t.y2 - t.y1)
    a2 = (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])
    return inter / (a1 + a2 - inter + 1e-6)


class MinimalTracker:
    """Inline temporal tracker — identical logic to HAWKEYE's TemporalTracker."""

    def __init__(self, confirm_frames=3, iou_threshold=0.25, max_miss=2):
        self.tracks        = []
        self.confirm_frames = confirm_frames
        self.iou_thresh    = iou_threshold
        self.max_miss      = max_miss

    def reset(self):
        self.tracks = []
        _Track._id  = 0

    def update(self, detections):
        matched_t = set(); matched_d = set()

        if self.tracks and detections:
            mat = np.zeros((len(self.tracks), len(detections)))
            for ti, t in enumerate(self.tracks):
                for di, d in enumerate(detections):
                    mat[ti, di] = _iou(t, d)
            while mat.max() >= self.iou_thresh:
                ti, di = np.unravel_index(mat.argmax(), mat.shape)
                self.tracks[ti].update(detections[di])
                matched_t.add(ti); matched_d.add(di)
                mat[ti, :] = -1; mat[:, di] = -1

        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.mark_missed()

        for di, d in enumerate(detections):
            if di not in matched_d:
                self.tracks.append(_Track(d, self.confirm_frames))

        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        return [t for t in self.tracks if t.confirmed]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    images_dir: Path,
    labels_dir: Path,
    preview_dir: Path,
    model,
    class_names: dict,
    extract_fps: float = 2.0,
    conf: float = 0.35,
    top_crop: float = 0.50,
    bot_crop: float = 0.05,
    confirm_frames: int = 3,
    iou_threshold: float = 0.25,
    max_miss: int = 2,
) -> dict:

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval   = max(1, round(src_fps / extract_fps))
    stem       = video_path.stem

    tracker = MinimalTracker(confirm_frames, iou_threshold, max_miss)

    print(f"\n{'─'*60}")
    print(f"  {video_path.name}")
    print(f"  {src_fps:.0f}fps | {total_fr} frames | "
          f"saving every {interval}th | HAWKEYE conf≥{conf} confirm={confirm_frames}f")

    saved = confirmed_total = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        y_start = int(h * top_crop)
        y_end   = int(h * (1.0 - bot_crop))
        cropped = frame[y_start:y_end, :]

        # YOLO on cropped region
        results = model.predict(cropped, conf=conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "conf": float(box.conf[0]),
                    "cls":  int(box.cls[0]),
                })

        confirmed = tracker.update(detections)

        # Save at extraction rate
        if frame_idx % interval == 0:
            fname    = f"{stem}_{frame_idx:06d}"
            img_path = images_dir  / f"{fname}.jpg"
            lbl_path = labels_dir  / f"{fname}.txt"
            prv_path = preview_dir / f"{fname}_preview.jpg"

            cv2.imwrite(str(img_path), frame)

            lines = []
            preview = frame.copy()

            for t in confirmed:
                # Convert cropped coords → full-frame coords
                fx1 = t.x1
                fy1 = t.y1 + y_start
                fx2 = t.x2
                fy2 = t.y2 + y_start

                # Clamp to frame bounds
                fx1 = max(0, min(fx1, w - 1))
                fy1 = max(0, min(fy1, h - 1))
                fx2 = max(0, min(fx2, w - 1))
                fy2 = max(0, min(fy2, h - 1))

                # YOLO normalised format
                cx  = ((fx1 + fx2) / 2) / w
                cy  = ((fy1 + fy2) / 2) / h
                bw  = (fx2 - fx1) / w
                bh  = (fy2 - fy1) / h
                cls = t.box.get("cls", 1)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                confirmed_total += 1

                # Draw on preview (green = HAWKEYE confirmed)
                cv2.rectangle(preview, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                label = f"[CONFIRMED] {class_names.get(cls, str(cls))} hits={t.hits}"
                cv2.putText(preview, label, (fx1, max(fy1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # Draw crop region indicator on preview
            cv2.line(preview, (0, y_start), (w, y_start), (255, 165, 0), 1)
            cv2.putText(preview, "YOLO region start", (4, y_start + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

            lbl_path.write_text("\n".join(lines))
            cv2.imwrite(str(prv_path), preview)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"  → {saved} frames saved | {confirmed_total} HAWKEYE-confirmed detections")

    return {
        "video":                    video_path.name,
        "frames_extracted":         saved,
        "hawkeye_confirmed_total":  confirmed_total,
        "avg_confirmed_per_frame":  round(confirmed_total / max(saved, 1), 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run HAWKEYE pipeline on new videos to generate high-quality pre-annotations"
    )
    parser.add_argument("--video_dir", required=True,
                        help='Folder containing fod_*.mp4 (e.g. "data/raw/new data")')
    parser.add_argument("--model", required=True,
                        help="Path to best.pt weights")
    parser.add_argument("--out", default="data/staging_hawkeye",
                        help="Output folder (default: data/staging_hawkeye)")
    parser.add_argument("--fps",            type=float, default=2.0)
    parser.add_argument("--conf",           type=float, default=0.35,
                        help="YOLO confidence threshold (default 0.35 — matches HAWKEYE config)")
    parser.add_argument("--confirm_frames", type=int,   default=3,
                        help="Frames needed to confirm FOD (default 3 — matches HAWKEYE config)")
    parser.add_argument("--top_crop",       type=float, default=0.50,
                        help="Fraction of frame cropped from top (default 0.50)")
    parser.add_argument("--bot_crop",       type=float, default=0.05,
                        help="Fraction of frame cropped from bottom (default 0.05)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Missing dependency. Run:  pip install ultralytics")

    model       = YOLO(args.model)
    class_names = model.names
    print(f"Model   : {args.model}")
    print(f"Classes : {class_names}")
    print(f"HAWKEYE : conf≥{args.conf} | confirm={args.confirm_frames} frames "
          f"| top_crop={args.top_crop} | bot_crop={args.bot_crop}")

    video_dir = Path(args.video_dir)
    out       = Path(args.out)
    img_dir   = out / "images"
    lbl_dir   = out / "labels"
    prv_dir   = out / "preview"
    for d in [img_dir, lbl_dir, prv_dir]:
        d.mkdir(parents=True, exist_ok=True)

    (out / "classes.txt").write_text("\n".join(class_names.values()))

    fod_videos = sorted(video_dir.glob("fod_*.mp4"))
    if not fod_videos:
        sys.exit(f"No fod_*.mp4 found in {video_dir}")

    summary = []
    for vp in fod_videos:
        stats = process_video(
            vp, img_dir, lbl_dir, prv_dir, model, class_names,
            extract_fps=args.fps, conf=args.conf,
            top_crop=args.top_crop, bot_crop=args.bot_crop,
            confirm_frames=args.confirm_frames,
        )
        summary.append(stats)

    total_frames = sum(s["frames_extracted"] for s in summary)
    total_conf   = sum(s["hawkeye_confirmed_total"] for s in summary)

    summary_obj = {
        "model":                   args.model,
        "pipeline":                "HAWKEYE (YOLO + temporal tracker)",
        "conf_threshold":          args.conf,
        "confirm_frames":          args.confirm_frames,
        "top_crop":                args.top_crop,
        "fps_extracted":           args.fps,
        "videos_processed":        len(summary),
        "total_frames_extracted":  total_frames,
        "total_confirmed_detections": total_conf,
        "per_video":               summary,
        "note": (
            "Detections here required the SAME bounding box to appear in "
            f">={args.confirm_frames} consecutive video frames. Far fewer "
            "false positives than staging_new/ (raw YOLO). Less cleanup "
            "work for the annotation team."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary_obj, indent=2))

    print(f"\n{'═'*60}")
    print(f"  DONE — HAWKEYE pre-annotation")
    print(f"  {total_frames} frames | {total_conf} confirmed detections")
    print(f"  Output: {out.resolve()}")
    print(f"{'═'*60}")
    print(f"\nCompare with staging_new/ (raw YOLO) to see how many FPs")
    print(f"the temporal tracker suppressed.\n")


if __name__ == "__main__":
    main()
