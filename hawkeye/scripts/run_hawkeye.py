"""
HAWKEYE — Full Pipeline Inference

Runs the three-component detection pipeline on a video file or live camera:
    1. Farneback optical flow + egomotion subtraction → flow candidates
    2. YOLOv8 (fine-tuned) on full frame → YOLO candidates
    3. PatchCore scoring per candidate → anomaly vote
    4. Fusion: 2-of-3 votes required → alert

Usage:
    python scripts/run_hawkeye.py
    python scripts/run_hawkeye.py --source data/raw/videos/test.mp4
    python scripts/run_hawkeye.py --source data/raw/videos/test.mp4 --visualise
    python scripts/run_hawkeye.py --source data/raw/videos/test.mp4 --speed 30

Options:
    --source      Video file path or 'camera' for live USB feed (default: config)
    --speed       Simulated vehicle speed in km/h (default: config)
    --visualise   Show live annotated video window
    --save        Save alert frames to outputs/alerts/
    --config      Config file path
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
from src.flow.residual import FlowResidual
from src.anomaly.patchcore import PatchCore
from src.fusion.hawkeye_fusion import HawkeyeFusion


def apply_roi_crop(frame: np.ndarray, top_crop: float, bot_crop: float) -> tuple:
    """
    Crop top and bottom from frame (removes vehicle hood and horizon).
    Returns cropped frame and y-offset for coordinate mapping.
    """
    h = frame.shape[0]
    y_start = int(h * top_crop)
    y_end = int(h * (1.0 - bot_crop))
    return frame[y_start:y_end, :], y_start


def main():
    parser = argparse.ArgumentParser(description="HAWKEYE inference pipeline")
    parser.add_argument("--source", type=str, default=None,
                        help="Video file path or 'camera'")
    parser.add_argument("--speed", type=float, default=None,
                        help="Simulated speed km/h")
    parser.add_argument("--visualise", action="store_true",
                        help="Show live video window")
    parser.add_argument("--save", action="store_true",
                        help="Save alert frames to outputs/alerts/")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger(
        "run_hawkeye",
        cfg.get("logging", "log_path", default="logs/hawkeye.log"),
        cfg.get("logging", "level", default="INFO")
    )

    logger.info("=" * 50)
    logger.info("HAWKEYE — Starting Inference")
    logger.info("=" * 50)

    # Override config with CLI args
    if args.source and args.source != "camera":
        cfg._cfg["camera"]["input_mode"] = "video_file"
        cfg._cfg["camera"]["video_file_path"] = args.source

    if args.speed:
        cfg._cfg["imu"]["simulated_speed_kmh"] = args.speed

    # Pipeline parameters
    top_crop = cfg.get("pipeline", "top_crop", default=0.22)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.15)
    warmup_frames = cfg.get("pipeline", "warmup_frames", default=30)

    # Output dirs
    alerts_dir = Path(cfg.get("outputs", "alerts_path", default="outputs/alerts"))
    detections_dir = Path(cfg.get("outputs", "detections_path", default="outputs/detections"))
    if args.save:
        alerts_dir.mkdir(parents=True, exist_ok=True)
        detections_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialise all components ──────────────────────────────
    logger.info("Loading components...")

    camera = CameraIngestion(cfg)
    camera.warmup(warmup_frames)

    yolo = YOLODetector(cfg)
    farneback = FarnebackFlow(cfg)
    egomotion = Egomotion(cfg)
    residual = FlowResidual(cfg)
    patchcore = PatchCore(cfg)
    fusion = HawkeyeFusion(cfg)

    logger.info("All components ready — pipeline running")

    # ── Main loop ──────────────────────────────────────────────
    frame_count = 0
    alert_count = 0
    fps_times = []

    try:
        while True:
            t_start = time.perf_counter()

            ret, frame = camera.read()
            if not ret:
                logger.info("End of stream")
                break

            # ROI crop
            cropped, y_offset = apply_roi_crop(frame, top_crop, bot_crop)

            # Component 1 — YOLO on full cropped frame
            yolo_detections = yolo.detect(cropped)

            # Component 2 — Farneback flow + egomotion residual
            flow = farneback.compute(cropped)
            if flow is None:
                frame_count += 1
                continue  # First frame — no flow yet

            expected_flow = egomotion.compute_expected_flow(
                cropped.shape[0], cropped.shape[1]
            )
            residual_map, anomaly_mask, flow_candidates = residual.compute(
                flow, expected_flow
            )

            # Component 3 + Fusion — PatchCore scoring + voting
            alerts = fusion.fuse(cropped, yolo_detections, flow_candidates, patchcore)

            # ── FPS tracking ───────────────────────────────────
            t_end = time.perf_counter()
            frame_ms = (t_end - t_start) * 1000
            fps_times.append(frame_ms)
            if len(fps_times) > 60:
                fps_times.pop(0)
            avg_fps = 1000.0 / (sum(fps_times) / len(fps_times))

            frame_count += 1

            if alerts:
                alert_count += len(alerts)
                logger.info(
                    f"Frame {frame_count:05d} — "
                    f"{len(alerts)} alert(s) — "
                    f"{avg_fps:.1f}fps — {frame_ms:.1f}ms"
                )

                if args.save:
                    annotated = fusion.visualise(cropped.copy(), alerts)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(str(alerts_dir / f"alert_{frame_count:05d}_{ts}.jpg"), annotated)

            if args.visualise:
                vis = fusion.visualise(cropped.copy(), alerts)
                # Status overlay
                status = (
                    f"Frame: {frame_count} | "
                    f"Alerts: {alert_count} | "
                    f"{avg_fps:.1f}fps | {frame_ms:.1f}ms"
                )
                cv2.putText(vis, status, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow("HAWKEYE", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User quit")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        camera.release()
        if args.visualise:
            cv2.destroyAllWindows()

    # ── Summary ────────────────────────────────────────────────
    avg_fps_overall = 1000.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0
    logger.info("=" * 50)
    logger.info(f"HAWKEYE — Run complete")
    logger.info(f"  Frames processed : {frame_count}")
    logger.info(f"  Total alerts     : {alert_count}")
    logger.info(f"  Avg FPS          : {avg_fps_overall:.1f}")
    logger.info(f"  Avg latency      : {1000/avg_fps_overall:.1f}ms" if avg_fps_overall > 0 else "")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
