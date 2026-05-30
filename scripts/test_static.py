"""
ARGUS-N Static Image Test
Tests the full pipeline on a single image.
No camera needed — just drop an image in and verify everything works.
Usage: PYTHONPATH=. python3 scripts/test_static.py --image path/to/image.jpg
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.flow.raft_flow import RAFTFlow
from src.flow.egomotion import Egomotion
from src.flow.residual import FlowResidual
from src.detection.yolo_detector import YOLODetector
from src.fusion.fusion import Fusion
from src.tracking.bytetrack_tracker import ByteTrackTracker


def parse_args():
    parser = argparse.ArgumentParser(description="ARGUS-N Static Image Test")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to test image"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=50.0,
        help="Simulated vehicle speed km/h"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output image to outputs/detections/"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ────────────────────────────────────────
    cfg = load_config()
    logger = get_logger(
        "test_static",
        cfg.get("logging", "log_path", default="logs/argus.log"),
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
    logger.info("Initialising pipeline modules...")
    raft      = RAFTFlow(cfg)
    egomotion = Egomotion(cfg)
    residual  = FlowResidual(cfg)
    detector  = YOLODetector(cfg)
    fusion    = Fusion(cfg)
    tracker   = ByteTrackTracker(cfg)

    # ── Simulate IMU ───────────────────────────────────────
    egomotion.update_imu(speed_kmh=args.speed)
    logger.info(f"IMU simulated at {args.speed} km/h")

    # ── RAFT needs two frames ──────────────────────────────
    # Feed same image twice to simulate T-1 and T
    # In real pipeline these are consecutive frames
    # For static test this gives zero residual on clean image
    # Drop a modified copy to simulate FOD presence
    logger.info("Computing optical flow (frame T-1 → frame T)...")

    # First frame — stores as T-1
    flow_t1 = raft.compute(frame)

    # Slightly shift frame to simulate vehicle motion
    # This creates a non-zero flow field for testing
    shift_px = int(
        (cfg.get("raft", "input_size", "height", default=1080) *
         (args.speed / 3.6)) /
        (cfg.get("camera", "fps", default=60) * 0.325)
    )
    shift_px = max(1, min(shift_px, 50))

    M = np.float32([[1, 0, 0], [0, 1, shift_px]])
    frame_shifted = cv2.warpAffine(frame, M, (w, h))

    logger.info(f"Simulated vehicle shift: {shift_px}px")

    # Second frame — compute flow
    flow = raft.compute(frame_shifted)

    if flow is None:
        logger.error("Flow computation returned None")
        return

    logger.info(f"Flow computed — shape: {flow.shape}")

    # ── Egomotion subtraction ──────────────────────────────
    expected_flow = egomotion.compute_expected_flow()
    logger.info(f"Expected flow computed — shape: {expected_flow.shape}")

    # ── Residual map ───────────────────────────────────────
    residual_map, anomaly_mask, candidates = residual.compute(flow, expected_flow)
    logger.info(f"Residual computed — {len(candidates)} candidate(s) found")

    for i, c in enumerate(candidates):
        logger.info(
            f"  Candidate {i+1}: "
            f"x={c['x']} y={c['y']} "
            f"w={c['w']} h={c['h']} "
            f"area={c['area']}px"
        )

    # ── YOLO detection ─────────────────────────────────────
    confirmed_detections, anomaly_frames = detector.detect(frame, candidates)
    logger.info(f"YOLO confirmed: {len(confirmed_detections)} detection(s)")

    for i, det in enumerate(confirmed_detections):
        logger.info(
            f"  Detection {i+1}: "
            f"{det['class_name']} "
            f"conf={det['confidence']:.2f} "
            f"box=({det['x1']},{det['y1']},{det['x2']},{det['y2']})"
        )

    # ── Fusion ─────────────────────────────────────────────
    fused = fusion.merge(candidates, confirmed_detections)
    logger.info(f"Fused detections: {len(fused)}")

    # ── Tracking ───────────────────────────────────────────
    conf_window = egomotion.get_dynamic_confirmation_window()
    confirmed_fods, active_tracks = tracker.update(fused, conf_window)
    logger.info(f"Active tracks: {len(active_tracks)}")
    logger.info(f"Confirmed FODs: {len(confirmed_fods)}")

    # ── Visualise ──────────────────────────────────────────
    vis = frame.copy()
    vis = residual.visualise(vis, candidates)
    vis = fusion.visualise(vis, fused)
    vis = tracker.visualise(vis, active_tracks)

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
        f"ARGUS-N Static Test | "
        f"Candidates: {len(candidates)} | "
        f"Detections: {len(confirmed_detections)} | "
        f"FOD: {len(confirmed_fods)}",
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
    cv2.imshow("ARGUS-N Static Test", vis)
    logger.info("Press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ── Summary ────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info(f"  Image          : {image_path.name}")
    logger.info(f"  Flow backend   : {'raft' if raft.use_raft else 'farneback'}")
    logger.info(f"  Vehicle speed  : {args.speed} km/h")
    logger.info(f"  Shift simulated: {shift_px}px")
    logger.info(f"  Candidates     : {len(candidates)}")
    logger.info(f"  YOLO detections: {len(confirmed_detections)}")
    logger.info(f"  Fused          : {len(fused)}")
    logger.info(f"  Confirmed FODs : {len(confirmed_fods)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
