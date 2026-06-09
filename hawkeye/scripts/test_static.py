"""
HAWKEYE Static Image Test
[Adapted from argusN/scripts/test_static.py]

Tests the full HAWKEYE pipeline on a single image.
No camera or video needed — drop an image in and verify everything works.

Changes from argusN original:
  - RAFTFlow         → FarnebackFlow (no weights needed)
  - original Fusion  → HawkeyeFusion (3-component voting)
  - ByteTrackTracker → removed (HAWKEYE has no tracking layer)
  - PatchCore        → added (third vote component)
  - egomotion.compute_expected_flow() now takes frame dims as args

Usage:
  PYTHONPATH=. python3 scripts/test_static.py --image path/to/image.jpg
  PYTHONPATH=. python3 scripts/test_static.py --image path/to/image.jpg --speed 30 --save
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.flow.farneback import FarnebackFlow
from src.flow.egomotion import Egomotion
from src.flow.residual import FlowResidual
from src.detection.yolo_detector import YOLODetector
from src.anomaly.patchcore import PatchCore
from src.fusion.hawkeye_fusion import HawkeyeFusion


def parse_args():
    parser = argparse.ArgumentParser(description="HAWKEYE Static Image Test")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to test image"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=30.0,
        help="Simulated vehicle speed km/h (default: 30)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output image to outputs/detections/"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Config file path"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ────────────────────────────────────────
    cfg = load_config(args.config)
    logger = get_logger(
        "test_static",
        cfg.get("logging", "log_path", default="logs/hawkeye.log"),
        cfg.get("logging", "level", default="INFO")
    )

    # ── Verify image exists ────────────────────────────────
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

    # ── Initialise modules ─────────────────────────────────
    logger.info("Initialising HAWKEYE pipeline modules...")
    farneback = FarnebackFlow(cfg)
    egomotion  = Egomotion(cfg)
    residual   = FlowResidual(cfg)
    detector   = YOLODetector(cfg)
    patchcore  = PatchCore(cfg)
    fusion     = HawkeyeFusion(cfg)

    # ── Simulate IMU ───────────────────────────────────────
    egomotion.update_imu(speed_kmh=args.speed)
    logger.info(f"IMU simulated at {args.speed} km/h")

    # ── Farneback needs two frames ─────────────────────────
    # Feed same image twice to simulate T-1 and T.
    # First call stores frame as T-1, second computes flow.
    # Slightly shift frame to simulate vehicle forward motion —
    # this creates a non-zero flow field for testing.
    logger.info("Computing optical flow (frame T-1 → frame T)...")

    fps = cfg.get("camera", "fps", default=60)
    camera_height_m = cfg.get("egomotion", "camera_height_m", default=0.325)
    focal_length_px = cfg.get("egomotion", "focal_length_px", default=1200.0)

    speed_ms = args.speed / 3.6
    displacement_m = speed_ms / fps
    shift_px = int((focal_length_px * displacement_m) / camera_height_m)
    shift_px = max(1, min(shift_px, 50))

    # First frame — stores as T-1
    farneback.compute(frame)

    # Shift frame to simulate forward motion
    M = np.float32([[1, 0, 0], [0, 1, shift_px]])
    frame_shifted = cv2.warpAffine(frame, M, (w, h))

    logger.info(f"Simulated vehicle shift: {shift_px}px")

    # Second frame — compute flow
    flow = farneback.compute(frame_shifted)

    if flow is None:
        logger.error("Flow computation returned None — unexpected")
        return

    logger.info(f"Flow computed — shape: {flow.shape}")

    # ── Egomotion subtraction ──────────────────────────────
    expected_flow = egomotion.compute_expected_flow(h, w)
    logger.info(f"Expected flow computed — shape: {expected_flow.shape}")

    # ── Residual map ───────────────────────────────────────
    residual_map, anomaly_mask, flow_candidates = residual.compute(flow, expected_flow)
    logger.info(f"Residual computed — {len(flow_candidates)} flow candidate(s)")

    for i, c in enumerate(flow_candidates):
        logger.info(
            f"  Flow candidate {i+1}: "
            f"x={c['x']} y={c['y']} "
            f"w={c['w']} h={c['h']} "
            f"area={c['area']}px"
        )

    # ── YOLO detection ─────────────────────────────────────
    yolo_detections = detector.detect(frame)
    logger.info(f"YOLO: {len(yolo_detections)} detection(s)")

    for i, det in enumerate(yolo_detections):
        logger.info(
            f"  Detection {i+1}: "
            f"{det['class_name']} "
            f"conf={det['confidence']:.2f} "
            f"box=({det['x1']},{det['y1']},{det['x2']},{det['y2']})"
        )

    # ── Fusion (YOLO + Flow + PatchCore voting) ────────────
    alerts = fusion.fuse(frame, yolo_detections, flow_candidates, patchcore)
    logger.info(f"Fusion: {len(alerts)} alert(s)")

    for i, alert in enumerate(alerts):
        logger.info(
            f"  Alert {i+1}: "
            f"votes={alert['votes']}/3 "
            f"(YOLO={alert['yolo_vote']} "
            f"Flow={alert['flow_vote']} "
            f"PC={alert['patchcore_vote']} score={alert['patchcore_score']:.3f}) "
            f"at ({alert['cx']:.0f},{alert['cy']:.0f})"
        )

    # ── Visualise ──────────────────────────────────────────
    vis = frame.copy()
    vis = residual.visualise(vis, flow_candidates)
    vis = fusion.visualise(vis, alerts)

    # Residual magnitude overlay — bottom left
    res_display = cv2.normalize(
        residual_map, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    res_color = cv2.applyColorMap(res_display, cv2.COLORMAP_JET)
    res_small = cv2.resize(res_color, (320, 180))
    vis[h-190:h-10, 10:330] = res_small
    cv2.putText(
        vis, "Flow Residual",
        (10, h - 195),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1
    )

    # Status overlay
    cv2.putText(
        vis,
        f"HAWKEYE Static Test | "
        f"Flow candidates: {len(flow_candidates)} | "
        f"YOLO: {len(yolo_detections)} | "
        f"Alerts: {len(alerts)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ── Save output ────────────────────────────────────────
    if args.save:
        out_path = Path("outputs/detections")
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"test_{image_path.stem}_result.jpg"
        cv2.imwrite(str(out_file), vis)
        logger.info(f"Output saved: {out_file}")

    # ── Show result ────────────────────────────────────────
    cv2.imshow("HAWKEYE Static Test", vis)
    logger.info("Press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ── Summary ────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info(f"  Image            : {image_path.name}")
    logger.info(f"  Flow backend     : farneback")
    logger.info(f"  Vehicle speed    : {args.speed} km/h")
    logger.info(f"  Shift simulated  : {shift_px}px")
    logger.info(f"  Flow candidates  : {len(flow_candidates)}")
    logger.info(f"  YOLO detections  : {len(yolo_detections)}")
    logger.info(f"  Fusion alerts    : {len(alerts)}")
    logger.info(f"  PatchCore ready  : {patchcore.memory_bank is not None}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
