"""
run_hawkeye.py — HAWKEYE live inference pipeline.

Pipeline:
  1. Ingest frame (video file or USB camera)
  2. Apply ROI crop
  3. YOLO inference
  4. TemporalTracker confirmation (>= confirm_frames consecutive detections)
  5. Draw confirmed FODs, save alert frames, display stream

Usage (run from inside hawkeye/ directory):
    python scripts/run_hawkeye.py --video path/to/video.mp4 --visualise
    python scripts/run_hawkeye.py --camera 0 --visualise
    python scripts/run_hawkeye.py --video path/to/video.mp4
"""

import cv2
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.ingestion.camera import CameraIngestion
from src.tracking.temporal_tracker import TemporalTracker


def apply_roi_crop(frame, top_frac, bot_frac):
    h, w = frame.shape[:2]
    y_start = int(h * top_frac)
    y_end   = int(h * (1.0 - bot_frac))
    return frame[y_start:y_end, :], y_start


def draw_confirmed(frame, confirmed, y_offset):
    RED = (0, 0, 255)
    for fod in confirmed:
        x1 = fod["x1"]
        y1 = fod["y1"] + y_offset
        x2 = fod["x2"]
        y2 = fod["y2"] + y_offset
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
        cv2.putText(frame, f"FOD {fod['confidence']:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
    if confirmed:
        cv2.putText(frame, f"ALERT: {len(confirmed)} FOD DETECTED",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 3)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Run HAWKEYE inference.")
    parser.add_argument("--config",    default="config/config.yaml")
    parser.add_argument("--video",     default=None)
    parser.add_argument("--camera",    type=int, default=None)
    parser.add_argument("--visualise", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = get_logger("run_hawkeye",
                     cfg.get("logging", "log_path", default="logs/hawkeye.log"),
                     cfg.get("logging", "level",    default="INFO"))

    if args.video:
        cfg._cfg["camera"]["input_mode"]      = "video_file"
        cfg._cfg["camera"]["video_file_path"] = args.video
    elif args.camera is not None:
        cfg._cfg["camera"]["input_mode"] = "usb"

    from ultralytics import YOLO
    model_path = cfg.get("yolo", "model_path")
    if not Path(model_path).exists():
        log.error(f"YOLO weights not found: {model_path}")
        sys.exit(1)

    yolo     = YOLO(model_path)
    tracker  = TemporalTracker(cfg)
    cam      = CameraIngestion(cfg, camera_index=args.camera or 0)
    cam.warmup(cfg.get("pipeline", "warmup_frames", default=10))

    top_crop  = cfg.get("pipeline", "top_crop", default=0.60)
    bot_crop  = cfg.get("pipeline", "bot_crop", default=0.05)
    imgsz     = cfg.get("yolo", "input_size",   default=640)
    conf_t    = cfg.get("yolo", "confidence_threshold", default=0.35)
    iou_t     = cfg.get("yolo", "iou_threshold",         default=0.45)
    device    = cfg.device
    alert_dir = Path(cfg.get("outputs", "alerts_path", default="outputs/alerts"))
    alert_dir.mkdir(parents=True, exist_ok=True)

    frame_idx   = 0
    alert_count = 0
    fps_timer   = time.time()
    fps_display = 0.0

    log.info("HAWKEYE running — press Q to quit")

    try:
        for frame in cam:
            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed     = time.time() - fps_timer
                fps_display = 30.0 / elapsed if elapsed > 0 else 0.0
                fps_timer   = time.time()

            cropped, y_offset = apply_roi_crop(frame, top_crop, bot_crop)

            results = yolo.predict(cropped, imgsz=imgsz, conf=conf_t,
                                   iou=iou_t, verbose=False, device=device)
            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "confidence": float(box.conf[0]),
                        "class_id": 0, "class_name": "fod"
                    })

            confirmed = tracker.update(detections)

            if confirmed:
                alert_count += 1
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                annotated = draw_confirmed(frame.copy(), confirmed, y_offset)
                cv2.imwrite(str(alert_dir / f"alert_{ts}_f{frame_idx:06d}.jpg"), annotated)
                log.info(f"FOD CONFIRMED frame {frame_idx} — {len(confirmed)} track(s)")

            if args.visualise:
                display = draw_confirmed(frame.copy(), confirmed, y_offset)
                h, w = frame.shape[:2]
                cv2.putText(display, f"FPS: {fps_display:.1f}",
                            (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("HAWKEYE", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cam.release()
        cv2.destroyAllWindows()

    log.info(f"Done — {frame_idx} frames, {alert_count} confirmed alert frames")


if __name__ == "__main__":
    main()
