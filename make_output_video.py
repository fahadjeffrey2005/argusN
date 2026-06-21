"""
make_output_video.py — Run a YOLO model on a video and save annotated output.

Applies the same ROI crop as HAWKEYE (top 50%, bottom 5%) before running YOLO
so detections are limited to the runway surface — not hangars, sky, or horizon.

Usage:
    python make_output_video.py --model best_v2.pt --video clean1.mp4
    python make_output_video.py --model best.pt    --video fod1.mp4 --out v1_fod.mp4

    # Disable ROI crop (see full frame detections):
    python make_output_video.py --model best_v2.pt --video clean1.mp4 --no_crop
"""

import argparse
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True,  help="Path to .pt weights")
    parser.add_argument("--video",     required=True,  help="Path to input video")
    parser.add_argument("--out",       default=None,   help="Output path (default: <model>_<video>.mp4)")
    parser.add_argument("--conf",      type=float, default=0.35)
    parser.add_argument("--top_crop",  type=float, default=0.50,
                        help="Fraction to crop from top — same as HAWKEYE (default 0.50)")
    parser.add_argument("--bot_crop",  type=float, default=0.05,
                        help="Fraction to crop from bottom (default 0.05)")
    parser.add_argument("--no_crop",   action="store_true",
                        help="Disable ROI crop and run on full frame")
    parser.add_argument("--fps_out",   type=float, default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    model_path = Path(args.model)
    video_path = Path(args.video)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_path = Path(args.out) if args.out else Path(f"{model_path.stem}_{video_path.stem}.mp4")

    model = YOLO(str(model_path))

    cap          = cv2.VideoCapture(str(video_path))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps      = args.fps_out or src_fps

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (w, h),
    )

    # ROI crop bounds
    if args.no_crop:
        y_start, y_end = 0, h
    else:
        y_start = int(h * args.top_crop)
        y_end   = int(h * (1.0 - args.bot_crop))

    print(f"\nModel     : {model_path.name}")
    print(f"Video     : {video_path.name}  ({total_frames} frames @ {src_fps:.0f}fps)")
    print(f"Output    : {out_path}")
    print(f"Conf      : {args.conf}")
    if args.no_crop:
        print(f"ROI crop  : DISABLED — full frame")
    else:
        print(f"ROI crop  : top {args.top_crop*100:.0f}% removed, "
              f"bottom {args.bot_crop*100:.0f}% removed  "
              f"(rows {y_start}–{y_end} of {h})")
    print(f"\nProcessing...")

    frame_idx        = 0
    total_dets       = 0
    frames_with_dets = 0
    t_start          = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped = frame[y_start:y_end, :]
        results  = model.predict(cropped, conf=args.conf, verbose=False)
        dets_this_frame = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                # Convert back to full-frame coordinates
                y1 += y_start
                y2 += y_start

                conf_val = float(box.conf[0])
                cls      = int(box.cls[0])
                label    = f"{model.names[cls]} {conf_val:.2f}"

                # RED boxes — easy to see
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
                dets_this_frame += 1
                total_dets += 1

        if dets_this_frame > 0:
            frames_with_dets += 1
            cv2.putText(frame, f"!! FOD DETECTED x{dets_this_frame} !!",
                        (12, y_start + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)

        # Draw ROI crop line so you can see exactly what the model sees
        if not args.no_crop:
            cv2.line(frame, (0, y_start), (w, y_start), (255, 165, 0), 1)
            cv2.line(frame, (0, y_end),   (w, y_end),   (255, 165, 0), 1)

        # HUD
        timestamp = frame_idx / src_fps
        mm, ss    = divmod(int(timestamp), 60)
        hud = f"Frame {frame_idx}/{total_frames}  |  {mm:02d}:{ss:02d}  |  Detections: {total_dets}"
        cv2.rectangle(frame, (0, 0), (w, 26), (20, 20, 20), -1)
        cv2.putText(frame, hud, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = 100 * frame_idx / total_frames
            print(f"  {pct:.0f}%  frame {frame_idx}/{total_frames}  dets so far: {total_dets}")

    cap.release()
    writer.release()

    duration_s = total_frames / src_fps
    fp_per_min = total_dets / (duration_s / 60) if duration_s > 0 else 0

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Total detections : {total_dets}")
    print(f"  Detection rate   : {fp_per_min:.2f} per minute")
    print(f"  Output           : {out_path.resolve()}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
