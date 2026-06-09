"""
PRIME Static Image Test
Tests the full PRIME pipeline on a single image.
No camera needed — drop an image in and verify everything runs end to end.

Usage:
    PYTHONPATH=. python3 scripts/test_static.py --image path/to/image.jpg --speed 30
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.flow.farneback import FarnebackFlow
from src.flow.bump_detector import BumpDetector
from src.flow.egomotion import Egomotion
from src.flow.residual import FlowResidual
from src.detection.yolo_detector import YOLODetector
from src.fusion.prime_fusion import PrimeFusion
from src.semantic.crop_builder import CropBuilder
from src.semantic.cnn_classifier import CNNClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="PRIME Static Image Test")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--speed", type=float, default=30.0, help="Simulated vehicle speed km/h")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--save", action="store_true", help="Save output image to outputs/detections/")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    logger = get_logger(
        "test_static",
        cfg.get("logging", "log_path", default="logs/prime.log"),
        cfg.get("logging", "level", default="INFO")
    )

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return

    logger.info(f"Loading image: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error("Failed to read image")
        return

    h, w = frame.shape[:2]
    logger.info(f"Image size: {w}x{h}")

    # ── Initialise pipeline ────────────────────────────────
    logger.info("Initialising PRIME pipeline modules...")
    flow_engine  = FarnebackFlow(cfg)
    bump_det     = BumpDetector(window=30, k=3.0)
    egomotion    = Egomotion(cfg)
    residual     = FlowResidual(cfg)
    yolo         = YOLODetector(cfg)
    fusion       = PrimeFusion(cfg)
    crop_builder = CropBuilder(cfg)
    classifier   = CNNClassifier(cfg)

    # ── Simulate IMU ───────────────────────────────────────
    egomotion.update_imu(speed_kmh=args.speed)
    logger.info(f"IMU simulated at {args.speed} km/h")

    # ── Farneback needs two frames ─────────────────────────
    # Feed image as T-1, shifted copy as T to simulate vehicle motion
    flow_t1 = flow_engine.compute(frame)   # stores frame, returns None

    fps = cfg.get("camera", "fps", default=60)
    cam_h = cfg.get("egomotion", "camera_height_m", default=0.325)
    focal = cfg.get("egomotion", "focal_length_px", default=1200.0)
    shift_px = int((focal * (args.speed / 3.6)) / (fps * cam_h))
    shift_px = max(1, min(shift_px, 50))

    M = np.float32([[1, 0, 0], [0, 1, shift_px]])
    frame_shifted = cv2.warpAffine(frame, M, (w, h))
    logger.info(f"Simulated vehicle shift: {shift_px}px")

    flow = flow_engine.compute(frame_shifted)
    if flow is None:
        logger.error("Flow computation returned None")
        return

    logger.info(f"Flow computed — shape: {flow.shape}")

    # ── Bump check ─────────────────────────────────────────
    is_bump = bump_det.update(flow)
    if is_bump:
        logger.warning("Bump detected — flow pathway discarded this frame")

    # ── Egomotion + residual ───────────────────────────────
    egomotion.update_imu(speed_kmh=args.speed)
    expected_flow = egomotion.compute_expected_flow()

    roi_h, roi_w = frame.shape[:2]
    expected_roi = cv2.resize(expected_flow, (roi_w, roi_h))

    flow_candidates = [] if is_bump else []
    if not is_bump:
        residual_map, anomaly_mask, flow_candidates = residual.compute(flow, expected_roi)
        logger.info(f"Residual computed — {len(flow_candidates)} flow candidate(s)")

    # ── YOLO ──────────────────────────────────────────────
    yolo_candidates, _ = yolo.detect(frame, flow_candidates)
    logger.info(f"YOLO candidates: {len(yolo_candidates)}")

    # ── Fusion (source tagging) ────────────────────────────
    merged = fusion.merge(yolo_candidates, flow_candidates)
    logger.info(f"Merged candidates: {len(merged)}")
    for c in merged:
        logger.info(f"  [{c['tag']}] box=({c['x1']},{c['y1']},{c['x2']},{c['y2']})")

    # ── 4-channel crop + CNN ───────────────────────────────
    flow_mag = flow_engine.flow_magnitude(flow)
    crops_and_candidates = crop_builder.build_batch(frame, flow_mag, merged)
    results = classifier.classify_batch(crops_and_candidates)

    fod_results = [(cls, cand) for cls, cand in results if cls["is_fod"]]
    logger.info(f"CNN classifications: {len(results)} total, {len(fod_results)} FOD")

    for cls, cand in results:
        logger.info(
            f"  [{cand['tag']}] → {cls['class_name']} "
            f"conf={cls['confidence']:.2f} "
            f"is_fod={cls['is_fod']}"
        )

    # ── Visualise ──────────────────────────────────────────
    vis = frame.copy()

    # Flow candidates — orange
    for c in flow_candidates:
        cv2.rectangle(vis, (c["x"], c["y"]), (c["x"]+c["w"], c["y"]+c["h"]), (0, 165, 255), 1)

    # CNN results
    for cls_result, cand in results:
        colour = (0, 0, 255) if cls_result["is_fod"] else (180, 180, 180)
        label = f"{cls_result['class_name']} {cls_result['confidence']:.2f} [{cand['tag']}]"
        cv2.rectangle(vis, (cand["x1"], cand["y1"]), (cand["x2"], cand["y2"]), colour, 2)
        cv2.putText(vis, label, (cand["x1"], max(0, cand["y1"]-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    # Status bar
    cv2.putText(
        vis,
        f"PRIME Static Test | Speed: {args.speed}km/h | "
        f"Merged: {len(merged)} | FOD alerts: {len(fod_results)}",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
    )

    if args.save:
        out_dir = Path(cfg.get("outputs", "detections_path", default="outputs/detections"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"test_{image_path.stem}_prime.jpg"
        cv2.imwrite(str(out_file), vis)
        logger.info(f"Output saved: {out_file}")

    cv2.imshow("PRIME Static Test", vis)
    logger.info("Press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ── Summary ────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("PRIME TEST SUMMARY")
    logger.info(f"  Image            : {image_path.name}")
    logger.info(f"  Vehicle speed    : {args.speed} km/h")
    logger.info(f"  Shift simulated  : {shift_px}px")
    logger.info(f"  Bump detected    : {is_bump}")
    logger.info(f"  Flow candidates  : {len(flow_candidates)}")
    logger.info(f"  YOLO candidates  : {len(yolo_candidates)}")
    logger.info(f"  Merged (fusion)  : {len(merged)}")
    logger.info(f"  CNN results      : {len(results)}")
    logger.info(f"  FOD alerts       : {len(fod_results)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
