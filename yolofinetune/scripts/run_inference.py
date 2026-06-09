"""
run_inference.py — Run YOLOFINETUNE live inference on video or USB camera.

Pipeline per frame:
  1. Ingest frame
  2. Apply ROI crop (remove top 22% hood, bottom 15% vehicle)
  3. Resize to 640x640 for YOLO
  4. Run fine-tuned YOLOv8n
  5. NMS + confidence filter
  6. If FOD found: draw boxes, save frame, log alert
  7. Display MJPEG stream (optional)

Usage:
    # Video file:
    python scripts/run_inference.py --video data/raw/videos/recording.mp4 --visualise

    # USB camera (live):
    python scripts/run_inference.py --camera 0 --visualise

    # No display (headless, saves frames only):
    python scripts/run_inference.py --video data/raw/videos/recording.mp4
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


def apply_roi_crop(frame, top_frac: float, bot_frac: float):
    """
    Remove top_frac from the top (vehicle hood) and bot_frac from the bottom.
    Returns cropped frame and the y-offset for remapping detections back.
    """
    h, w = frame.shape[:2]
    y_start = int(h * top_frac)
    y_end   = int(h * (1.0 - bot_frac))
    return frame[y_start:y_end, :], y_start


def remap_boxes_to_original(boxes, y_offset: int, crop_h: int, orig_h: int, orig_w: int):
    """
    Translate bounding boxes from cropped-frame coordinates back to original frame.
    boxes: list of (x1, y1, x2, y2, conf, cls) in pixel coords of crop.
    """
    remapped = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        remapped.append((x1, y1 + y_offset, x2, y2 + y_offset, conf, cls))
    return remapped


def draw_detections(frame, boxes, alert_count: int):
    """Draw bounding boxes and alert overlay onto frame."""
    for (x1, y1, x2, y2, conf, cls) in boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        label = f"FOD {conf:.2f}"
        cv2.putText(
            frame, label,
            (int(x1), int(y1) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    if boxes:
        cv2.putText(
            frame, f"ALERT: {len(boxes)} FOD DETECTED",
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
        )

    return frame


def run_inference(cfg_path: str = "config/config.yaml", args=None):
    from ultralytics import YOLO

    cfg = load_config(cfg_path)
    log = get_logger(
        "inference",
        cfg.get("logging", "log_path", default="logs/yolofinetune.log"),
        cfg.get("logging", "level", default="INFO")
    )

    model_path  = cfg.get("yolo", "model_path")
    conf_thresh = cfg.get("yolo", "confidence_threshold", default=0.35)
    iou_thresh  = cfg.get("yolo", "iou_threshold", default=0.45)
    imgsz       = cfg.get("yolo", "input_size", default=640)
    top_crop    = cfg.get("pipeline", "top_crop", default=0.22)
    bot_crop    = cfg.get("pipeline", "bot_crop", default=0.15)
    warmup      = cfg.get("pipeline", "warmup_frames", default=30)
    det_path    = Path(cfg.get("outputs", "detections_path", default="outputs/detections"))
    alert_path  = Path(cfg.get("outputs", "alerts_path", default="outputs/alerts"))

    det_path.mkdir(parents=True, exist_ok=True)
    alert_path.mkdir(parents=True, exist_ok=True)

    # Override source from CLI if provided
    if args and args.video:
        cfg._cfg["camera"]["input_mode"] = "video_file"
        cfg._cfg["camera"]["video_file_path"] = args.video
    elif args and args.camera is not None:
        cfg._cfg["camera"]["input_mode"] = "usb"

    if not Path(model_path).exists():
        log.error(f"Model weights not found: {model_path}")
        log.error("Run train_yolo.py first.")
        sys.exit(1)

    log.info("Loading YOLO model...")
    model = YOLO(model_path)

    log.info("Initialising camera ingestion...")
    cam = CameraIngestion(cfg, camera_index=args.camera if args and args.camera else 0)
    cam.warmup(warmup)

    visualise = args and args.visualise
    frame_idx  = 0
    alert_count = 0
    fps_timer  = time.time()
    fps_display = 0.0

    log.info("Inference running — press Q to quit")

    try:
        for frame in cam:
            frame_idx += 1

            # FPS calculation (rolling over 30 frames)
            if frame_idx % 30 == 0:
                elapsed = time.time() - fps_timer
                fps_display = 30.0 / elapsed if elapsed > 0 else 0.0
                fps_timer = time.time()

            # ROI crop
            cropped, y_offset = apply_roi_crop(frame, top_crop, bot_crop)
            crop_h, crop_w = cropped.shape[:2]

            # YOLO inference
            results = model(
                cropped,
                imgsz=imgsz,
                conf=conf_thresh,
                iou=iou_thresh,
                verbose=False
            )

            # Parse detections
            boxes = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls  = int(box.cls[0])
                    boxes.append((x1, y1, x2, y2, conf, cls))

            # Remap to original frame coordinates
            if boxes:
                boxes = remap_boxes_to_original(boxes, y_offset, crop_h, frame.shape[0], frame.shape[1])
                alert_count += 1

                # Save alert frame
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                alert_file = alert_path / f"alert_{ts}_f{frame_idx:06d}.jpg"
                annotated = draw_detections(frame.copy(), boxes, alert_count)
                cv2.imwrite(str(alert_file), annotated)
                log.info(f"FOD ALERT frame {frame_idx} — {len(boxes)} detection(s) → {alert_file.name}")

            # Visualise
            if visualise:
                display = draw_detections(frame.copy(), boxes, alert_count)
                cv2.putText(
                    display, f"FPS: {fps_display:.1f}",
                    (frame.shape[1] - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.imshow("YOLOFINETUNE — FOD Detection", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("User quit")
                    break

    finally:
        cam.release()
        cv2.destroyAllWindows()

    log.info(f"Inference complete — {frame_idx} frames processed, {alert_count} alerts")


def main():
    parser = argparse.ArgumentParser(description="Run YOLOFINETUNE inference.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--video", default=None, help="Path to video file (overrides config)")
    parser.add_argument("--camera", type=int, default=None, help="USB camera index")
    parser.add_argument("--visualise", action="store_true", help="Show live display window")
    args = parser.parse_args()

    run_inference(cfg_path=args.config, args=args)


if __name__ == "__main__":
    main()
