"""
make_demo_video.py — Run PRIME on FOD footage and write annotated output video.

PRIME pipeline:
  1. YOLO detects candidates every frame
  2. Farneback flow → magnitude map (4th CNN channel)
  3. CNN classifies candidates → 5 classes (fod/shadow/marking/strobe/clean)
  4. Only FOD-classified candidates → TemporalTracker
  5. Confirmed tracks drawn as red boxes

Output: outputs/demo_prime_fod1.mp4
Comparable to:
  yolofinetune/outputs/demo_yolofinetune_fod1.mp4
  hawkeye/outputs/demo_hawkeye_fod1.mp4

Usage (run from inside prime/ directory):
    python scripts/make_demo_video.py \\
        --video ../yolofinetune/data/raw/videos/fod_sessions/fod1.mp4 \\
        --output outputs/demo_prime_fod1.mp4 \\
        --preview
"""

import cv2
import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.flow.farneback import FarnebackFlow
from src.semantic.crop_builder import CropBuilder
from src.semantic.cnn_classifier import CNNClassifier
from src.tracking.temporal_tracker import TemporalTracker


def apply_roi_crop(frame, top_frac: float, bot_frac: float):
    h, w = frame.shape[:2]
    y_start = int(h * top_frac)
    y_end   = int(h * (1.0 - bot_frac))
    return frame[y_start:y_end, :], y_start


def draw_confirmed(frame, confirmed: list, y_offset: int):
    """Draw confirmed FOD tracks on full frame. Matches hawkeye style exactly."""
    RED = (0, 0, 255)
    for fod in confirmed:
        x1 = fod["x1"]
        y1 = fod["y1"] + y_offset
        x2 = fod["x2"]
        y2 = fod["y2"] + y_offset
        conf = fod["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
        cv2.putText(frame, f"FOD {conf:.2f}",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2)
    if confirmed:
        cv2.putText(frame, f"ALERT: {len(confirmed)} FOD DETECTED",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 3)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Render PRIME demo video.")
    parser.add_argument("--video",   required=True)
    parser.add_argument("--output",  default="outputs/demo_prime_fod1.mp4")
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = get_logger("make_demo_video",
                     cfg.get("logging", "log_path", default="logs/prime.log"),
                     cfg.get("logging", "level",    default="INFO"))

    video_path = Path(args.video)
    if not video_path.exists():
        log.error(f"Video not found: {video_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_crop = cfg.get("pipeline", "top_crop", default=0.50)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.05)
    imgsz    = cfg.get("yolo", "input_size",   default=640)
    conf_t   = cfg.get("yolo", "confidence_threshold", default=0.28)
    iou_t    = cfg.get("yolo", "iou_threshold",         default=0.45)
    device   = cfg.device

    # Probe video
    cap_probe    = cv2.VideoCapture(str(video_path))
    source_fps   = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    log.info("=" * 55)
    log.info("PRIME — Demo Video")
    log.info("=" * 55)
    log.info(f"Input  : {video_path.name}  ({orig_w}x{orig_h} @ {source_fps:.0f}fps)")
    log.info(f"Output : {output_path}")
    log.info(f"ROI    : top={top_crop}, bot={bot_crop}")
    log.info(f"Confirm: {cfg.get('tracker','confirm_frames',default=3)} frames")

    # VideoWriter — output is full frame (matches hawkeye style)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, source_fps, (orig_w, orig_h))

    # Components
    from ultralytics import YOLO
    yolo       = YOLO(cfg.get("yolo", "model_path", default="models/yolo/finetuned/best.pt"))
    farneback  = FarnebackFlow(cfg)
    crop_bld   = CropBuilder(cfg)
    classifier = CNNClassifier(cfg)
    tracker    = TemporalTracker(cfg)

    if args.preview:
        cv2.namedWindow("PRIME", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PRIME", min(orig_w, 1280), min(orig_h, 720))

    cap         = cv2.VideoCapture(str(video_path))
    frame_idx   = 0
    alert_count = 0
    fps_timer   = time.time()
    fps_display = 0.0

    log.info("Rendering...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed     = time.time() - fps_timer
                fps_display = 30.0 / elapsed if elapsed > 0 else 0.0
                fps_timer   = time.time()

            # ROI crop
            cropped, y_offset = apply_roi_crop(frame, top_crop, bot_crop)
            roi_h, roi_w = cropped.shape[:2]

            # YOLO
            results = yolo.predict(cropped, imgsz=imgsz, conf=conf_t,
                                   iou=iou_t, verbose=False, device=device)
            dets = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dets.append({
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "confidence": float(box.conf[0]),
                    })

            # Flow magnitude for CNN 4th channel
            flow     = farneback.compute(cropped)
            flow_mag = (farneback.magnitude_map(flow) if flow is not None
                        else np.zeros((roi_h, roi_w), dtype=np.float32))

            # CNN classify
            crops   = crop_bld.build_batch(cropped, flow_mag, dets)
            clf_res = classifier.classify_batch(crops)

            # Filter to FOD only
            fod_dets = []
            for cls_r, cand in clf_res:
                if cls_r["is_fod"]:
                    fod_dets.append({
                        "x1": cand["x1"], "y1": cand["y1"],
                        "x2": cand["x2"], "y2": cand["y2"],
                        "confidence": cls_r["confidence"],
                    })

            # Temporal confirmation
            confirmed = tracker.update(fod_dets)
            if confirmed:
                alert_count += 1

            # Draw on full frame (same style as hawkeye)
            out_frame = frame.copy()
            out_frame = draw_confirmed(out_frame, confirmed, y_offset)
            cv2.putText(out_frame, f"FPS: {fps_display:.1f}",
                        (orig_w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(out_frame, "PRIME",
                        (20, orig_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            writer.write(out_frame)

            if args.preview:
                dw = min(orig_w, 1280)
                dh = int(orig_h * dw / orig_w)
                cv2.imshow("PRIME", cv2.resize(out_frame, (dw, dh)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % 200 == 0:
                pct = frame_idx / total_frames * 100
                log.info(f"  {frame_idx}/{total_frames} ({pct:.0f}%) — alert frames: {alert_count}")

    finally:
        cap.release()
        writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    log.info("=" * 55)
    log.info(f"Done — {frame_idx} frames rendered, {alert_count} alert frames")
    log.info(f"Output : {output_path}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
