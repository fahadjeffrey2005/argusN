"""
PRIME — run_prime.py
Full pipeline inference: YOLO + Farneback flow + CNN classifier.
Reads from video file or USB camera, draws detections, saves alert frames.

Usage:
    python scripts/run_prime.py --config config/config.yaml [--visualise] [--save]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.ingestion.camera import CameraIngestion
from src.detection.yolo_detector import YOLODetector
from src.flow.farneback import FarnebackFlow
from src.flow.egomotion import Egomotion
from src.flow.residual import FlowResidual
from src.fusion.prime_fusion import PrimeFusion
from src.semantic.crop_builder import CropBuilder
from src.semantic.cnn_classifier import CNNClassifier


def draw_fod_detections(frame: np.ndarray, fod_results: list) -> np.ndarray:
    vis = frame.copy()
    for classification, candidate in fod_results:
        if not classification["is_fod"]:
            continue
        x1, y1, x2, y2 = candidate["x1"], candidate["y1"], candidate["x2"], candidate["y2"]
        tag = candidate.get("tag", "")
        conf = classification["confidence"]
        label = f"FOD [{tag}] {conf:.2f}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            vis, label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
        )
    return vis


def main():
    parser = argparse.ArgumentParser(description="PRIME full pipeline inference")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--visualise", action="store_true", help="Show live preview window")
    parser.add_argument("--save", action="store_true", help="Save alert frames to outputs/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger(
        "run_prime",
        cfg.get("logging", "log_path", default="logs/prime.log"),
        cfg.get("logging", "level", default="INFO")
    )

    # ── Initialise components ───────────────────────────────────────
    camera = CameraIngestion(cfg)
    yolo = YOLODetector(cfg)
    flow_engine = FarnebackFlow(cfg)
    ego = Egomotion(cfg)
    residual = FlowResidual(cfg)
    fusion = PrimeFusion(cfg)
    crop_builder = CropBuilder(cfg)
    classifier = CNNClassifier(cfg)

    detections_dir = Path(cfg.get("outputs", "detections_path", default="outputs/detections"))
    alerts_dir = Path(cfg.get("outputs", "alerts_path", default="outputs/alerts"))
    if args.save:
        detections_dir.mkdir(parents=True, exist_ok=True)
        alerts_dir.mkdir(parents=True, exist_ok=True)

    top_crop = cfg.get("pipeline", "top_crop", default=0.22)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.15)
    warmup = cfg.get("pipeline", "warmup_frames", default=30)

    camera.warmup(warmup)

    # ── Pipeline loop ──────────────────────────────────────────────
    frame_idx = 0
    alert_count = 0
    t_start = time.time()

    logger.info("PRIME pipeline running — press q or ESC to stop")

    for frame in camera:
        h = frame.shape[0]
        y_start = int(h * top_crop)
        y_end = int(h * (1 - bot_crop))
        frame_roi = frame[y_start:y_end, :]

        t0 = time.time()

        # Optical flow
        raw_flow = flow_engine.compute(frame_roi)
        if raw_flow is None:
            frame_idx += 1
            continue

        flow_mag = flow_engine.flow_magnitude(raw_flow)

        # Egomotion expected flow
        expected = ego.compute_expected_flow()
        roi_h, roi_w = frame_roi.shape[:2]
        expected_roi = cv2.resize(expected, (roi_w, roi_h))

        # Flow residual candidates
        _, _, flow_candidates = residual.compute(raw_flow, expected_roi)

        # YOLO candidates
        yolo_candidates = yolo.detect(frame_roi)

        # Fusion
        merged = fusion.merge(yolo_candidates, flow_candidates)

        # Build 4-channel crops
        crops_and_candidates = crop_builder.build_batch(frame_roi, flow_mag, merged)

        # Classify
        results = classifier.classify_batch(crops_and_candidates)

        # Filter FODs
        fod_results = [(c, cand) for c, cand in results if c["is_fod"]]

        latency_ms = (time.time() - t0) * 1000
        elapsed = time.time() - t_start
        fps = frame_idx / max(elapsed, 0.001)

        if fod_results:
            alert_count += 1
            logger.info(
                f"Frame {frame_idx:06d} | "
                f"FOD ALERT x{len(fod_results)} | "
                f"{latency_ms:.1f}ms | {fps:.1f}fps"
            )
            if args.save:
                vis = draw_fod_detections(frame_roi, fod_results)
                out_path = alerts_dir / f"alert_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), vis)

        if args.visualise:
            vis = draw_fod_detections(frame_roi, fod_results)
            # Overlay stats
            cv2.putText(vis, f"FPS: {fps:.1f}  Latency: {latency_ms:.1f}ms  Alerts: {alert_count}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("PRIME — FOD Detection", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

        frame_idx += 1

    cv2.destroyAllWindows()
    total_time = time.time() - t_start
    avg_fps = frame_idx / max(total_time, 0.001)

    logger.info(
        f"\nPipeline finished.\n"
        f"  Frames processed : {frame_idx}\n"
        f"  Total alerts     : {alert_count}\n"
        f"  Avg FPS          : {avg_fps:.1f}\n"
        f"  Total time       : {total_time:.1f}s"
    )


if __name__ == "__main__":
    main()
