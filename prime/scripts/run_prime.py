"""
PRIME — Full Pipeline Inference

Pipeline:
    1. YOLO detects candidates on ROI-cropped frame
    2. Farneback flow → magnitude map (4th CNN channel)
    3. CNN classifies each YOLO candidate → 5 classes
    4. Only FOD-classified candidates enter the temporal tracker
    5. Tracker confirms FODs present for >= confirm_frames consecutive frames → ALERT

NOTE: On static footage the flow channel is near-zero.
The CNN is trained to be robust with near-zero flow (treats it as clean_tarmac signal).

Usage (from inside prime/):
    python scripts/run_prime.py
    python scripts/run_prime.py --source ../yolofinetune/data/raw/videos/fod_sessions/fod1.mp4
    python scripts/run_prime.py --source ... --visualise --speed 30
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.ingestion.camera import CameraIngestion
from src.detection.yolo_detector import YOLODetector
from src.flow.farneback import FarnebackFlow
from src.flow.egomotion import Egomotion
from src.semantic.crop_builder import CropBuilder
from src.semantic.cnn_classifier import CNNClassifier
from src.tracking.temporal_tracker import TemporalTracker


def apply_roi(frame, top_crop, bot_crop):
    h = frame.shape[0]
    y0 = int(h * top_crop)
    y1 = int(h * (1.0 - bot_crop))
    return frame[y0:y1, :], y0


def main():
    parser = argparse.ArgumentParser(description="PRIME inference pipeline")
    parser.add_argument("--source",    default=None)
    parser.add_argument("--speed",     type=float, default=None)
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--save",      action="store_true")
    parser.add_argument("--config",    default="config/config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    logger = get_logger("run_prime",
                        cfg.get("logging", "log_path", default="logs/prime.log"),
                        cfg.get("logging", "level",    default="INFO"))

    if args.source and args.source != "camera":
        cfg._cfg["camera"]["input_mode"]      = "video_file"
        cfg._cfg["camera"]["video_file_path"] = args.source
    if args.speed:
        cfg._cfg["imu"]["simulated_speed_kmh"] = args.speed

    top_crop = cfg.get("pipeline", "top_crop",      default=0.50)
    bot_crop = cfg.get("pipeline", "bot_crop",      default=0.05)
    warmup   = cfg.get("pipeline", "warmup_frames", default=30)

    alerts_dir = Path(cfg.get("outputs", "alerts_path", default="outputs/alerts"))
    if args.save:
        alerts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("PRIME — Starting Inference")
    logger.info("=" * 50)

    # ── Components ─────────────────────────────────────────
    camera     = CameraIngestion(cfg)
    camera.warmup(warmup)
    yolo       = YOLODetector(cfg)
    farneback  = FarnebackFlow(cfg)
    egomotion  = Egomotion(cfg)
    crop_bld   = CropBuilder(cfg)
    classifier = CNNClassifier(cfg)
    tracker    = TemporalTracker(cfg)

    logger.info("All components ready")

    frame_count = alert_count = 0
    fps_times   = []

    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = camera.read()
            if not ret:
                logger.info("End of stream")
                break

            roi, y_offset = apply_roi(frame, top_crop, bot_crop)
            roi_h, roi_w  = roi.shape[:2]

            # Step 1 — YOLO candidates
            yolo_dets = yolo.detect(roi)

            # Step 2 — Flow magnitude for CNN 4th channel
            flow = farneback.compute(roi)
            if flow is not None:
                flow_mag = farneback.magnitude_map(flow)
            else:
                flow_mag = np.zeros((roi_h, roi_w), dtype=np.float32)

            # Step 3 — CNN: classify each YOLO candidate
            # Build 4-channel crops from YOLO detections directly
            crops   = crop_bld.build_batch(roi, flow_mag, yolo_dets)
            results = classifier.classify_batch(crops)

            # Step 4 — Filter: keep only CNN-confirmed FOD candidates
            fod_candidates = []
            for cls_result, candidate in results:
                if cls_result["is_fod"]:
                    fod_candidates.append({
                        "x1": candidate["x1"], "y1": candidate["y1"],
                        "x2": candidate["x2"], "y2": candidate["y2"],
                        "confidence": cls_result["confidence"],
                        "cnn_class": cls_result["class_name"],
                    })

            # Step 5 — Temporal confirmation
            confirmed = tracker.update(fod_candidates)

            t1       = time.perf_counter()
            frame_ms = (t1 - t0) * 1000
            fps_times.append(frame_ms)
            if len(fps_times) > 60:
                fps_times.pop(0)
            avg_fps = 1000.0 / (sum(fps_times) / len(fps_times))

            frame_count += 1

            if confirmed:
                alert_count += 1
                logger.info(
                    f"Frame {frame_count:05d} — "
                    f"{len(confirmed)} FOD confirmed — "
                    f"{avg_fps:.1f}fps — {frame_ms:.1f}ms"
                )
                if args.save:
                    vis = roi.copy()
                    for fod in confirmed:
                        cv2.rectangle(vis, (fod["x1"], fod["y1"]),
                                      (fod["x2"], fod["y2"]), (0, 0, 255), 2)
                        cv2.putText(vis, f"FOD {fod['confidence']:.2f}",
                                    (fod["x1"], max(0, fod["y1"]-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(str(alerts_dir / f"alert_{frame_count:05d}_{ts}.jpg"), vis)

            if args.visualise:
                vis = roi.copy()
                for fod in confirmed:
                    cv2.rectangle(vis, (fod["x1"], fod["y1"]),
                                  (fod["x2"], fod["y2"]), (0, 0, 255), 2)
                    cv2.putText(vis, f"FOD {fod['confidence']:.2f}",
                                (fod["x1"], max(0, fod["y1"]-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                if confirmed:
                    cv2.putText(vis, f"ALERT: {len(confirmed)} FOD",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(
                    vis,
                    f"Frame: {frame_count} | Alerts: {alert_count} | "
                    f"{avg_fps:.1f}fps | {frame_ms:.1f}ms",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                cv2.imshow("PRIME", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        camera.release()
        if args.visualise:
            cv2.destroyAllWindows()

    avg_fps_f = (1000.0 / (sum(fps_times) / len(fps_times))) if fps_times else 0
    logger.info("=" * 50)
    logger.info(f"PRIME — Run complete")
    logger.info(f"  Frames    : {frame_count}")
    logger.info(f"  Alerts    : {alert_count}")
    logger.info(f"  Avg FPS   : {avg_fps_f:.1f}")
    logger.info(f"  Avg ms    : {1000/avg_fps_f:.1f}" if avg_fps_f > 0 else "")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
