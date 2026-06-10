"""
make_demo_video.py — Run trained YOLOFINETUNE on a video and produce an annotated output video.

For visual comparison across Model 1 (YOLOFINETUNE), Model 2 (HAWKEYE), Model 3 (PRIME).
Draws bounding boxes, confidence scores, alert counter, FPS, and ROI crop lines.

Usage:
    python scripts/make_demo_video.py \
        --video data/raw/videos/fod_sessions/fod1.mp4 \
        --output outputs/demo_yolofinetune_fod1.mp4

    # Without ROI crop (full frame):
    python scripts/make_demo_video.py \
        --video data/raw/videos/fod_sessions/fod1.mp4 \
        --output outputs/demo_yolofinetune_fod1.mp4 \
        --no-crop
"""

import cv2
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

# ── Overlay colours ─────────────────────────────────────────
COL_BOX      = (0, 0, 255)       # red — FOD bounding box
COL_LABEL    = (255, 255, 255)   # white text
COL_ALERT    = (0, 0, 220)       # red alert banner
COL_FPS      = (0, 220, 0)       # green FPS
COL_CLEAN    = (0, 200, 0)       # green — no detection
COL_CROP     = (180, 180, 0)     # yellow — ROI crop lines
COL_MODEL    = (255, 200, 0)     # cyan-ish — model label


def draw_roi_lines(frame, top_frac: float, bot_frac: float):
    """Draw semi-transparent ROI boundary lines on frame."""
    h, w = frame.shape[:2]
    y_top = int(h * top_frac)
    y_bot = int(h * (1.0 - bot_frac))
    cv2.line(frame, (0, y_top), (w, y_top), COL_CROP, 1, cv2.LINE_AA)
    cv2.line(frame, (0, y_bot), (w, y_bot), COL_CROP, 1, cv2.LINE_AA)
    cv2.putText(frame, "ROI", (8, y_top - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_CROP, 1)
    return frame


def draw_frame(frame, boxes, frame_idx, alert_count, fps, model_label="YOLOFINETUNE"):
    """Draw all overlays onto frame. Returns annotated copy."""
    h, w = frame.shape[:2]
    out = frame.copy()

    # Model label — top left
    cv2.putText(out, f"Model: {model_label}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_MODEL, 2)

    # Frame counter — top left below model
    cv2.putText(out, f"Frame: {frame_idx:06d}",
                (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_LABEL, 1)

    # FPS — top right
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(out, fps_text,
                (w - tw - 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_FPS, 2)

    # Alert counter — top right below FPS
    alert_text = f"Alerts: {alert_count}"
    (tw2, _), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(out, alert_text,
                (w - tw2 - 12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_LABEL, 1)

    if boxes:
        # Alert banner
        cv2.rectangle(out, (0, h - 48), (w, h), (0, 0, 180), -1)
        banner = f"  FOD DETECTED — {len(boxes)} object(s)  |  Total alerts: {alert_count}"
        cv2.putText(out, banner,
                    (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Bounding boxes
        for (x1, y1, x2, y2, conf, _cls) in boxes:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), COL_BOX, 2)
            label = f"FOD {conf:.2f}"
            lx, ly = int(x1), max(int(y1) - 8, 14)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(out, (lx, ly - lh - 2), (lx + lw + 4, ly + 2), COL_BOX, -1)
            cv2.putText(out, label, (lx + 2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_LABEL, 2)
    else:
        # Clean status bar
        cv2.rectangle(out, (0, h - 36), (w, h), (0, 80, 0), -1)
        cv2.putText(out, "  No FOD detected",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

    return out


def make_demo_video(
    video_path: str,
    output_path: str,
    cfg_path: str = "config/config.yaml",
    apply_crop: bool = True
):
    from ultralytics import YOLO

    cfg = load_config(cfg_path)
    log = get_logger(
        "demo",
        cfg.get("logging", "log_path", default="logs/yolofinetune.log"),
        cfg.get("logging", "level", default="INFO")
    )

    model_path  = cfg.get("yolo", "model_path")
    conf_thresh = cfg.get("yolo", "confidence_threshold", default=0.35)
    iou_thresh  = cfg.get("yolo", "iou_threshold", default=0.45)
    imgsz       = cfg.get("yolo", "input_size", default=640)
    top_crop    = cfg.get("pipeline", "top_crop", default=0.22) if apply_crop else 0.0
    bot_crop    = cfg.get("pipeline", "bot_crop", default=0.15) if apply_crop else 0.0

    if not Path(model_path).exists():
        log.error(f"Model not found: {model_path}. Run training first.")
        sys.exit(1)

    if not Path(video_path).exists():
        log.error(f"Video not found: {video_path}")
        sys.exit(1)

    log.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        sys.exit(1)

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (src_w, src_h))

    log.info(f"Input  : {video_path} ({src_w}x{src_h} @ {src_fps:.1f}fps, {total_frames} frames)")
    log.info(f"Output : {output_path}")
    log.info(f"ROI crop: top={top_crop:.0%}  bot={bot_crop:.0%}")

    frame_idx   = 0
    alert_count = 0
    fps_display = 0.0
    t_batch     = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # FPS rolling average over 30 frames
        if frame_idx % 30 == 0:
            elapsed = time.perf_counter() - t_batch
            fps_display = 30.0 / elapsed if elapsed > 0 else 0.0
            t_batch = time.perf_counter()

        # ROI crop for inference only
        h, w = frame.shape[:2]
        y_start = int(h * top_crop)
        y_end   = int(h * (1.0 - bot_crop))
        cropped = frame[y_start:y_end, :]

        # Inference
        results = model(cropped, imgsz=imgsz, conf=conf_thresh, iou=iou_thresh, verbose=False)

        # Parse detections — remap y coords back to full frame
        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                boxes.append((x1, y1 + y_start, x2, y2 + y_start, conf, cls))

        if boxes:
            alert_count += 1

        # Draw ROI lines + overlays onto full frame
        annotated = frame.copy()
        if apply_crop:
            draw_roi_lines(annotated, top_crop, bot_crop)
        annotated = draw_frame(annotated, boxes, frame_idx, alert_count, fps_display)

        writer.write(annotated)

        if frame_idx % 100 == 0:
            pct = 100 * frame_idx / total_frames if total_frames > 0 else 0
            log.info(f"  {frame_idx}/{total_frames} frames ({pct:.0f}%)  alerts so far: {alert_count}")

    cap.release()
    writer.release()

    log.info(f"\nDone.")
    log.info(f"  Total frames  : {frame_idx}")
    log.info(f"  Total alerts  : {alert_count}")
    log.info(f"  Alert rate    : {alert_count/frame_idx*100:.1f}% of frames")
    log.info(f"  Output video  : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Produce annotated demo video from YOLOFINETUNE.")
    parser.add_argument("--video",   required=True, help="Input video path")
    parser.add_argument("--output",  required=True, help="Output .mp4 path")
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--no-crop", action="store_true", help="Disable ROI crop")
    args = parser.parse_args()

    make_demo_video(
        video_path=args.video,
        output_path=args.output,
        cfg_path=args.config,
        apply_crop=not args.no_crop
    )


if __name__ == "__main__":
    main()
