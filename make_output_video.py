"""
make_output_video.py — Run a YOLO model on a video and save annotated output.

Use this to visually verify model performance before committing to new weights.
Run on clean1.mp4 to check false positive rate, or on a FOD video to check recall.

Usage:
    python make_output_video.py --model best_v2.pt --video clean1.mp4
    python make_output_video.py --model best_v2.pt --video clean1.mp4 --conf 0.35
    python make_output_video.py --model best.pt    --video clean1.mp4 --out original_output.mp4

    # Compare old vs new side by side:
    python make_output_video.py --model best.pt    --video clean1.mp4 --out v1_clean.mp4
    python make_output_video.py --model best_v2.pt --video clean1.mp4 --out v2_clean.mp4
"""

import argparse
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Generate annotated output video from YOLO model")
    parser.add_argument("--model",   required=True,  help="Path to .pt weights")
    parser.add_argument("--video",   required=True,  help="Path to input video")
    parser.add_argument("--out",     default=None,   help="Output video path (default: <model>_<video>.mp4)")
    parser.add_argument("--conf",    type=float, default=0.35, help="Confidence threshold (default 0.35)")
    parser.add_argument("--fps_out", type=float, default=None, help="Output FPS (default: same as input)")
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

    cap = cv2.VideoCapture(str(video_path))
    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = args.fps_out or src_fps

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (w, h),
    )

    print(f"\nModel   : {model_path.name}")
    print(f"Video   : {video_path.name}  ({total_frames} frames @ {src_fps:.0f}fps)")
    print(f"Output  : {out_path}")
    print(f"Conf    : {args.conf}")
    print(f"\nProcessing...")

    frame_idx    = 0
    total_dets   = 0
    frames_with_dets = 0
    t_start      = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=args.conf, verbose=False)
        dets_this_frame = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf_val = float(box.conf[0])
                cls      = int(box.cls[0])
                label    = f"{model.names[cls]} {conf_val:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                dets_this_frame += 1
                total_dets += 1

        if dets_this_frame > 0:
            frames_with_dets += 1
            cv2.putText(frame, f"DETECTION x{dets_this_frame}", (12, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)

        # HUD — top left
        elapsed    = time.time() - t_start
        proc_fps   = frame_idx / elapsed if elapsed > 0 else 0
        timestamp  = frame_idx / src_fps
        mm, ss     = divmod(int(timestamp), 60)
        hud = f"Frame {frame_idx}/{total_frames}  |  {mm:02d}:{ss:02d}  |  Total detections: {total_dets}"
        cv2.rectangle(frame, (0, 0), (w, 26), (20, 20, 20), -1)
        cv2.putText(frame, hud, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = 100 * frame_idx / total_frames
            print(f"  {pct:.0f}%  frame {frame_idx}/{total_frames}  "
                  f"detections so far: {total_dets}  ({proc_fps:.1f} fps)")

    cap.release()
    writer.release()

    duration_s  = total_frames / src_fps
    fp_per_min  = total_dets / (duration_s / 60) if duration_s > 0 else 0

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Frames processed  : {frame_idx}")
    print(f"  Total detections  : {total_dets}")
    print(f"  Frames with dets  : {frames_with_dets}")
    print(f"  FP rate           : {fp_per_min:.2f} per minute  "
          f"(if this is a clean video)")
    print(f"  Output saved      : {out_path.resolve()}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
