"""
compare_models.py — Run YOLOfinetune and HAWKEYE on the same video side by side.
==================================================================================

Produces two output videos and a printed comparison table:
  - yolofinetune_<video>.mp4  : raw YOLO + ROI crop, no temporal filtering
  - hawkeye_<video>.mp4       : YOLO + temporal tracker (3-frame confirmation)

The difference you see on screen IS the value HAWKEYE adds.

Usage (from argusN/ root):
    python compare_models.py --model yolofinetune/models/yolo/finetuned/best.pt \\
                             --video yolofinetune/data/raw/videos/fod1.mp4

    python compare_models.py --model yolofinetune/models/yolo/finetuned/best.pt \\
                             --video yolofinetune/data/raw/videos/clean1.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


# ── Inline minimal tracker (same logic as hawkeye/src/tracking/temporal_tracker.py) ──

class _Track:
    _id = 0
    def __init__(self, det, cf):
        _Track._id += 1
        self.tid = _Track._id
        self.box = det
        self.hits = 1
        self.miss = 0
        self.confirmed = False
        self.cf = cf

    def update(self, det):
        self.box = det; self.hits += 1; self.miss = 0
        if self.hits >= self.cf: self.confirmed = True

    def mark_missed(self): self.miss += 1

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
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    return inter / ((t.x2-t.x1)*(t.y2-t.y1) + (d["x2"]-d["x1"])*(d["y2"]-d["y1"]) - inter + 1e-6)


class Tracker:
    def __init__(self, confirm=3, iou=0.25, max_miss=2):
        self.tracks = []; self.cf = confirm; self.iou = iou; self.mm = max_miss

    def reset(self): self.tracks = []; _Track._id = 0

    def update(self, dets):
        mt = set(); md = set()
        if self.tracks and dets:
            mat = np.zeros((len(self.tracks), len(dets)))
            for ti, t in enumerate(self.tracks):
                for di, d in enumerate(dets): mat[ti,di] = _iou(t, d)
            while mat.max() >= self.iou:
                ti, di = np.unravel_index(mat.argmax(), mat.shape)
                self.tracks[ti].update(dets[di]); mt.add(ti); md.add(di)
                mat[ti,:] = -1; mat[:,di] = -1
        for ti, t in enumerate(self.tracks):
            if ti not in mt: t.mark_missed()
        for di, d in enumerate(dets):
            if di not in md: self.tracks.append(_Track(d, self.cf))
        self.tracks = [t for t in self.tracks if t.miss <= self.mm]
        return [t for t in self.tracks if t.confirmed]


# ── Main ──────────────────────────────────────────────────────────────────────

def process(video_path, model, out_path, top_crop, bot_crop, conf, use_tracker, tracker=None):
    cap = cv2.VideoCapture(str(video_path))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y_start = int(h * top_crop)
    y_end   = int(h * (1.0 - bot_crop))

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

    label   = "HAWKEYE" if use_tracker else "YOLOfinetune"
    color   = (255, 100, 0) if use_tracker else (0, 0, 255)   # blue for HAWKEYE, red for YOLO

    total_dets = 0; frames_with_dets = 0; frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        cropped = frame[y_start:y_end, :]
        results  = model.predict(cropped, conf=conf, verbose=False)
        raw_dets = []
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
                raw_dets.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                                  "conf":float(box.conf[0]),"cls":int(box.cls[0])})

        if use_tracker:
            confirmed = tracker.update(raw_dets)
            draw_dets = [{"x1":t.x1,"y1":t.y1,"x2":t.x2,"y2":t.y2,
                          "conf":t.box["conf"],"cls":t.box["cls"]} for t in confirmed]
        else:
            draw_dets = raw_dets

        for d in draw_dets:
            x1,y1,x2,y2 = d["x1"], d["y1"]+y_start, d["x2"], d["y2"]+y_start
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{model.names[d['cls']]} {d['conf']:.2f}",
                        (x1, max(y1-6,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            total_dets += 1

        if draw_dets:
            frames_with_dets += 1
            cv2.putText(frame, f"!! {label} DETECTION x{len(draw_dets)} !!",
                        (12, y_start+40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

        # ROI lines
        cv2.line(frame, (0,y_start), (w,y_start), (255,165,0), 1)
        cv2.line(frame, (0,y_end),   (w,y_end),   (255,165,0), 1)

        # HUD
        ts = frame_idx / src_fps; mm, ss = divmod(int(ts), 60)
        hud = f"{label}  |  Frame {frame_idx}/{total_frames}  |  {mm:02d}:{ss:02d}  |  Dets: {total_dets}"
        cv2.rectangle(frame, (0,0), (w,26), (20,20,20), -1)
        cv2.putText(frame, hud, (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release(); writer.release()

    duration_s  = total_frames / src_fps
    dets_per_min = total_dets / (duration_s / 60) if duration_s > 0 else 0

    return {
        "total_frames":   frame_idx,
        "duration_s":     round(duration_s, 1),
        "total_dets":     total_dets,
        "frames_with_dets": frames_with_dets,
        "dets_per_min":   round(dets_per_min, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--video",    required=True)
    parser.add_argument("--conf",     type=float, default=0.35)
    parser.add_argument("--top_crop", type=float, default=0.50)
    parser.add_argument("--bot_crop", type=float, default=0.05)
    parser.add_argument("--confirm",  type=int,   default=3,
                        help="HAWKEYE confirm frames (default 3)")
    args = parser.parse_args()

    from ultralytics import YOLO
    model     = YOLO(args.model)
    video_path = Path(args.video)
    stem      = video_path.stem

    print(f"\nModel  : {args.model}")
    print(f"Video  : {video_path}  (conf={args.conf})")
    print(f"ROI    : top {args.top_crop*100:.0f}% cropped")
    print(f"HAWKEYE confirm : {args.confirm} consecutive frames\n")

    # ── Run 1: YOLOfinetune (raw YOLO, no tracker) ──
    print("[ 1/2 ] Running YOLOfinetune (raw YOLO)...")
    yolo_out = Path(f"yolofinetune_{stem}.mp4")
    yolo_stats = process(video_path, model, yolo_out,
                         args.top_crop, args.bot_crop, args.conf,
                         use_tracker=False)
    print(f"        Done → {yolo_out}")

    # ── Run 2: HAWKEYE (YOLO + temporal tracker) ──
    print("[ 2/2 ] Running HAWKEYE (YOLO + temporal tracker)...")
    hawk_out = Path(f"hawkeye_{stem}.mp4")
    tracker  = Tracker(confirm=args.confirm)
    hawk_stats = process(video_path, model, hawk_out,
                         args.top_crop, args.bot_crop, args.conf,
                         use_tracker=True, tracker=tracker)
    print(f"        Done → {hawk_out}\n")

    # ── Print comparison table ──
    W = 58
    print("=" * W)
    print(f"  COMPARISON — {stem}")
    print("=" * W)
    print(f"  {'Metric':<28} {'YOLOfinetune':>12} {'HAWKEYE':>12}")
    print(f"  {'-'*28} {'-'*12} {'-'*12}")
    print(f"  {'Total detections':<28} {yolo_stats['total_dets']:>12} {hawk_stats['total_dets']:>12}")
    print(f"  {'Frames with detection':<28} {yolo_stats['frames_with_dets']:>12} {hawk_stats['frames_with_dets']:>12}")
    print(f"  {'Detections / minute':<28} {yolo_stats['dets_per_min']:>12.1f} {hawk_stats['dets_per_min']:>12.1f}")
    suppressed = yolo_stats['total_dets'] - hawk_stats['total_dets']
    pct = 100 * suppressed / max(yolo_stats['total_dets'], 1)
    print(f"  {'Detections suppressed':<28} {'-':>12} {f'{suppressed} ({pct:.0f}%)':>12}")
    print("=" * W)
    print(f"  Videos: {yolo_out}  |  {hawk_out}")
    print("=" * W)


if __name__ == "__main__":
    main()
