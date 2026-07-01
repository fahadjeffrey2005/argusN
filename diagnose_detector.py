"""
diagnose_detector.py
---------------------
Raw YOLO detections on fod1.mp4 with NO pipeline gates.
Shows exactly what the model sees at different confidence thresholds.
Extracts frames where something is detected for manual inspection.

Run from argusN/:
    python diagnose_detector.py --video fod1.mp4
    python diagnose_detector.py --video fod1.mp4 --conf 0.1   # very sensitive
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",   required=True)
    p.add_argument("--weights", default="yolofinetune/models/yolo/finetuned/best.pt")
    p.add_argument("--conf",    type=float, default=0.1,  help="Raw confidence threshold")
    p.add_argument("--device",  default="cuda")
    p.add_argument("--out",     default="runs/diagnose",  help="Output folder for annotated frames")
    p.add_argument("--max-frames", type=int, default=500, help="Only check first N frames")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names  # {0: 'bolt', 1: 'foreign object', ...}

    print(f"\n{'='*55}")
    print(f"  Raw Detector Diagnostic")
    print(f"{'='*55}")
    print(f"  Video  : {args.video}")
    print(f"  Weights: {args.weights}")
    print(f"  Conf   : {args.conf}  (NO pipeline gates)")
    print(f"  Classes: {names}")
    print(f"  Output : {out}/")
    print(f"{'='*55}\n")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.video}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {total} frames @ {fps:.0f}fps  {w}x{h}\n")

    # Per-class detection counts
    class_counts = {i: 0 for i in names}
    frames_with_fod = []   # frames where bolt/foreign object detected
    frame_idx = 0
    saved = 0

    while frame_idx < (args.max_frames or total):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, device=args.device, verbose=False)
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            cls_ids  = boxes.cls.cpu().numpy().astype(int)
            confs    = boxes.conf.cpu().numpy()
            xyxy     = boxes.xyxy.cpu().numpy().astype(int)

            # Count per class
            for c in cls_ids:
                if c in class_counts:
                    class_counts[c] += 1

            # Check if any real FOD classes detected (bolt=0, foreign object=1)
            fod_mask = (cls_ids == 0) | (cls_ids == 1)
            has_real_fod = fod_mask.any()

            # Save annotated frame for manual review (save every 10th detection frame, always save FOD)
            should_save = has_real_fod or (saved < 20 and frame_idx % 30 == 0 and len(cls_ids) > 0)

            if should_save:
                vis = frame.copy()
                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                    cls_id = cls_ids[i]
                    conf   = confs[i]
                    label  = names.get(cls_id, str(cls_id))
                    # Real FOD = red box, everything else = grey
                    color = (0, 0, 255) if cls_id in (0, 1) else (128, 128, 128)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, f"{label} {conf:.2f}", (x1, max(y1-6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                ts = frame_idx / fps
                mm, ss = divmod(int(ts), 60)
                tag = "FOD" if has_real_fod else "other"
                fname = f"frame_{frame_idx:05d}_{mm:02d}m{ss:02d}s_{tag}.jpg"
                cv2.imwrite(str(out / fname), vis)
                saved += 1

                if has_real_fod:
                    fod_confs = confs[fod_mask]
                    fod_classes = [names[c] for c in cls_ids[fod_mask]]
                    frames_with_fod.append((frame_idx, mm, ss, list(zip(fod_classes, fod_confs.tolist()))))

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{min(args.max_frames or total, total)} frames...", flush=True)

    cap.release()

    print(f"\n{'='*55}")
    print(f"  RESULTS (conf >= {args.conf}, no gates)")
    print(f"{'='*55}")
    print(f"\n  Detections per class:")
    for cls_id, name in names.items():
        n = class_counts.get(cls_id, 0)
        bar = "█" * min(n // 5, 40)
        flag = "  ← REAL FOD" if cls_id in (0, 1) else ""
        print(f"    {name:20s}: {n:5d}  {bar}{flag}")

    print(f"\n  Frames where bolt/foreign object detected: {len(frames_with_fod)}")
    if frames_with_fod:
        print(f"  Sample detections:")
        for fi, mm, ss, dets in frames_with_fod[:10]:
            print(f"    frame {fi:5d} ({mm:02d}:{ss:02d}) — {dets}")
    else:
        print(f"\n  !! ZERO real FOD detections at conf={args.conf}")
        print(f"  !! Model cannot see real FODs in this video at all.")
        print(f"  !! Root cause: training data doesn't match this camera setup.")
        print(f"  !! Fix: annotate frames from fod1.mp4 directly and retrain.")

    print(f"\n  Saved {saved} annotated frames to: {out}/")
    print(f"  RED boxes = bolt/foreign object")
    print(f"  GREY boxes = other classes (runway line etc.)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
