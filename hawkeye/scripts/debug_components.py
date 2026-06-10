"""
debug_components.py — Debug PatchCore and flow components.

Checks:
  1. PatchCore bank loading and scoring calibration
  2. Flow residual on a real video frame
  3. Per-vote breakdown on a sample frame from fod1.mp4

Usage (run from inside hawkeye/ directory):
    python scripts/debug_components.py
    python scripts/debug_components.py --video path/to/fod1.mp4
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.anomaly.patchcore import PatchCore
from src.flow.farneback import FarnebackFlow
from src.flow.egomotion import Egomotion
from src.flow.residual import FlowResidual
from src.detection.yolo_detector import YOLODetector
from src.fusion.hawkeye_fusion import HawkeyeFusion


def debug_patchcore(cfg, log):
    log.info("=" * 55)
    log.info("DEBUG — PatchCore")
    log.info("=" * 55)

    pc = PatchCore(cfg)

    if pc.memory_bank is None:
        log.error("Bank NOT loaded — check models/patchcore/bank.pt exists")
        return
    log.info(f"Bank loaded: {pc.memory_bank.shape[0]} vectors, dim={pc.memory_bank.shape[1]}")

    # Score random noise patch (should be anomalous vs clean tarmac)
    dummy_noise = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    score_noise = pc.score(dummy_noise)
    log.info(f"Random noise patch score : {score_noise:.4f}")

    # Score a black patch (very different from tarmac)
    dummy_black = np.zeros((128, 128, 3), dtype=np.uint8)
    score_black = pc.score(dummy_black)
    log.info(f"Black patch score        : {score_black:.4f}")

    # Score a grey patch (similar to tarmac)
    dummy_grey = np.full((128, 128, 3), 128, dtype=np.uint8)
    score_grey = pc.score(dummy_grey)
    log.info(f"Grey patch score         : {score_grey:.4f}")

    log.info(f"Anomaly threshold        : {pc.anomaly_threshold}")
    log.info(f"Noise flagged anomalous  : {score_noise >= pc.anomaly_threshold}")
    log.info(f"Black flagged anomalous  : {score_black >= pc.anomaly_threshold}")

    # Show raw L2 distances for calibration
    import torch
    feat = pc._extract_features(dummy_noise)
    diffs = pc.memory_bank - feat.unsqueeze(0)
    dists = torch.norm(diffs, dim=1)
    log.info(f"Min L2 distance (noise)  : {dists.min().item():.4f}")
    log.info(f"Max L2 distance (noise)  : {dists.max().item():.4f}")
    log.info(f"Mean L2 distance (noise) : {dists.mean().item():.4f}")


def debug_flow(cfg, log, video_path: str = None):
    log.info("=" * 55)
    log.info("DEBUG — Optical Flow")
    log.info("=" * 55)

    farneback = FarnebackFlow(cfg)
    egomotion  = Egomotion(cfg)
    residual   = FlowResidual(cfg)

    top_crop = cfg.get("pipeline", "top_crop", default=0.22)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.15)

    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(60):          # read first 60 frames
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            y_start = int(h * top_crop)
            y_end   = int(h * (1.0 - bot_crop))
            frames.append(frame[y_start:y_end, :])
        cap.release()
        log.info(f"Loaded {len(frames)} frames from {Path(video_path).name}")
    else:
        # Synthetic moving frames
        log.info("No video provided — using synthetic frames")
        H, W = int(1080 * (1.0 - top_crop - bot_crop)), 1920
        base = np.random.randint(100, 160, (H, W, 3), dtype=np.uint8)
        M   = np.float32([[1, 0, 0], [0, 1, 5]])
        frames = [base] + [cv2.warpAffine(base, np.float32([[1,0,0],[0,1,i*3]]), (W, H))
                           for i in range(1, 10)]

    # Prime flow
    farneback.compute(frames[0])
    egomotion.update_imu(speed_kmh=cfg.get("imu", "simulated_speed_kmh", default=30.0))

    total_candidates = 0
    for i, frame in enumerate(frames[1:], 1):
        ch, cw = frame.shape[:2]
        flow = farneback.compute(frame)
        if flow is None:
            continue
        exp  = egomotion.compute_expected_flow(ch, cw)
        residual_map, _, candidates = residual.compute(flow, exp)
        total_candidates += len(candidates)
        if i <= 5:
            log.info(f"  Frame {i:02d}: {len(candidates)} flow candidates, "
                     f"residual max={residual_map.max():.2f} mean={residual_map.mean():.2f}")

    log.info(f"Total candidates over {len(frames)-1} frames: {total_candidates}")
    log.info(f"Avg candidates/frame: {total_candidates / max(1, len(frames)-1):.1f}")
    log.info(f"Residual threshold: {cfg.get('flow', 'residual_threshold', default=2.5)}")


def debug_fusion_on_frame(cfg, log, video_path: str = None):
    log.info("=" * 55)
    log.info("DEBUG — Full Fusion (single frame)")
    log.info("=" * 55)

    from ultralytics import YOLO

    model_path = cfg.get("yolo", "model_path", default="models/yolo/finetuned/best.pt")
    if not Path(model_path).exists():
        log.error(f"YOLO weights not found: {model_path}")
        return

    yolo      = YOLO(model_path)
    farneback = FarnebackFlow(cfg)
    egomotion = Egomotion(cfg)
    residual  = FlowResidual(cfg)
    pc        = PatchCore(cfg)
    fusion    = HawkeyeFusion(cfg)

    top_crop = cfg.get("pipeline", "top_crop", default=0.22)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.15)
    conf     = cfg.get("yolo", "confidence_threshold", default=0.35)
    iou      = cfg.get("yolo", "iou_threshold",        default=0.45)
    imgsz    = cfg.get("yolo", "input_size",           default=640)

    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(100):
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            y_start = int(h * top_crop)
            y_end   = int(h * (1.0 - bot_crop))
            frames.append(frame[y_start:y_end, :])
        cap.release()
    else:
        log.warning("No video — skipping fusion debug")
        return

    egomotion.update_imu(speed_kmh=cfg.get("imu", "simulated_speed_kmh", default=30.0))

    log.info("Scanning video for first YOLO detection (skipping empty frames)...")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_frame = None
    found = 0
    frame_idx = 0

    while found < 10:
        ret, raw = cap.read()
        if not ret:
            break
        frame_idx += 1

        h, w = raw.shape[:2]
        y_start = int(h * top_crop)
        y_end   = int(h * (1.0 - bot_crop))
        frame   = raw[y_start:y_end, :]
        ch, cw  = frame.shape[:2]

        # YOLO
        results = yolo.predict(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        yolo_dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                yolo_dets.append({"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),
                                   "confidence":float(box.conf[0]),"class_id":0,"class_name":"fod"})

        # Flow (need two frames)
        flow = farneback.compute(frame)
        cands = []
        if flow is not None:
            exp = egomotion.compute_expected_flow(ch, cw)
            _, _, cands = residual.compute(flow, exp)

        # Only report frames where YOLO fires
        if not yolo_dets:
            continue

        found += 1

        # PatchCore score on first detection
        pc_score = None
        import torch
        if pc.memory_bank is not None:
            det   = yolo_dets[0]
            patch = frame[max(0,det["y1"]):det["y2"], max(0,det["x1"]):det["x2"]]
            if patch.size > 0:
                pc_score = pc.score(patch)
                # Also print raw L2 distance for calibration
                feat  = pc._extract_features(patch)
                diffs = pc.memory_bank - feat.unsqueeze(0)
                dists = torch.norm(diffs, dim=1)
                min_dist = dists.min().item()

        alerts = fusion.fuse(frame, yolo_dets, cands, pc)

        log.info(
            f"Video frame {frame_idx}/{total} | "
            f"YOLO={len(yolo_dets)} | "
            f"Flow cands={len(cands)} | "
            f"PC_score={f'{pc_score:.4f}' if pc_score is not None else 'N/A'} | "
            f"Min_L2={f'{min_dist:.2f}' if pc_score is not None else 'N/A'} | "
            f"PC_vote={1 if pc_score is not None and pc_score >= pc.anomaly_threshold else 0} | "
            f"Alerts={len(alerts)}"
        )

    cap.release()
    log.info(f"PatchCore threshold : {pc.anomaly_threshold}")
    log.info(f"Votes required      : {cfg.get('fusion','votes_required',default=2)}")
    log.info("--- Calibration hint ---")
    log.info("If Min_L2 >> 10, the scale=10 in patchcore.py is too small → scores near 1 always")
    log.info("If Min_L2 << 10, the scale=10 is too large → scores near 0 always")


def main():
    parser = argparse.ArgumentParser(description="Debug HAWKEYE components.")
    parser.add_argument("--video",  default=None, help="Path to fod1.mp4 for realistic test")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = get_logger("debug", cfg.get("logging","log_path",default="logs/hawkeye.log"),
                     cfg.get("logging","level",default="INFO"))

    debug_patchcore(cfg, log)
    debug_flow(cfg, log, args.video)
    if args.video:
        debug_fusion_on_frame(cfg, log, args.video)


if __name__ == "__main__":
    main()
