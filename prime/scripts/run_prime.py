"""
PRIME — Full Pipeline Inference

Pipeline:
    1. YOLO detects candidates on ROI-cropped frame
    2. Geometric filter — aspect ratio + area ratio (free, no calibration)
    3. Frame-diff magnitude appended as 4th CNN channel (one subtraction)
    4. Local contrast normalisation inside CropBuilder (runway-agnostic)
    5. CNN classifies each candidate → 5 classes
       - conf < 0.25   → hard reject
       - conf 0.25-0.90 → slow path (2-frame prefilter → 3-frame tracker)
       - conf > 0.90   → fast path (direct to 3-frame tracker)
    6. Only FOD-classified candidates enter the trackers
    7. Main tracker confirms FOD present >= 3 consecutive frames → ALERT

Usage (from inside prime/):
    python scripts/run_prime.py
    python scripts/run_prime.py --source path/to/fod1.mp4
    python scripts/run_prime.py --source path/to/fod1.mp4 --visualise
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
from src.flow.frame_diff import FrameDiff
from src.semantic.crop_builder import CropBuilder
from src.semantic.cnn_classifier import CNNClassifier
from src.tracking.temporal_tracker import TemporalTracker

# ── Geometric pre-filter (no calibration) ────────────────────────────────────
MAX_ASPECT_RATIO   = 4.0    # compact objects only
MAX_FRAME_AREA_FRAC = 0.05  # < 5% of frame area


def apply_roi(frame, top_crop, bot_crop):
    h = frame.shape[0]
    y0 = int(h * top_crop)
    y1 = int(h * (1.0 - bot_crop))
    return frame[y0:y1, :], y0


def passes_geometric_filter(det: dict, frame_h: int, frame_w: int) -> bool:
    w = max(1, det["x2"] - det["x1"])
    h = max(1, det["y2"] - det["y1"])
    if max(w / h, h / w) > MAX_ASPECT_RATIO:
        return False
    if (w * h) / max(1, frame_h * frame_w) > MAX_FRAME_AREA_FRAC:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="PRIME inference pipeline")
    parser.add_argument("--source",    default=None)
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

    top_crop = cfg.get("pipeline", "top_crop",      default=0.50)
    bot_crop = cfg.get("pipeline", "bot_crop",      default=0.05)
    warmup   = cfg.get("pipeline", "warmup_frames", default=30)

    alerts_dir = Path(cfg.get("outputs", "alerts_path", default="outputs/alerts"))
    if args.save:
        alerts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("PRIME — Starting Inference")
    logger.info("=" * 50)

    # ── Components ─────────────────────────────────────────────────────────
    camera     = CameraIngestion(cfg)
    camera.warmup(warmup)
    yolo       = YOLODetector(cfg)
    frame_diff = FrameDiff()
    crop_bld   = CropBuilder(cfg)
    classifier = CNNClassifier(cfg)

    # Two-stage tracker: prefilter (2 frames) + main (3 frames)
    prefilter  = TemporalTracker(cfg, confirm_frames_override=
                                 cfg.get("tracker", "prefilter_frames", default=2))
    tracker    = TemporalTracker(cfg)   # uses confirm_frames from config (default 3)

    logger.info(f"CNN        : reject < {classifier.conf_threshold}, "
                f"fast_path > {classifier.fast_path_threshold}")
    logger.info(f"Prefilter  : {prefilter.confirm_frames} frames")
    logger.info(f"Tracker    : {tracker.confirm_frames} frames")
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

            # Step 2 — Geometric filter (free, no ML)
            yolo_dets = [d for d in yolo_dets
                         if passes_geometric_filter(d, roi_h, roi_w)]

            # Step 3 — Frame-diff 4th channel
            diff_mag = frame_diff.compute(roi)

            # Step 4 — Build 4-channel crops (local contrast normalisation inside)
            crops = crop_bld.build_batch(roi, diff_mag, yolo_dets)

            # Step 5 — CNN classify
            clf_results = classifier.classify_batch(crops)

            # Step 6 — Route by confidence
            fast_cands = []   # CNN conf > 0.90 → direct to 3-frame tracker
            slow_cands = []   # CNN conf 0.25-0.90 → 2-frame prefilter first

            for cls_result, candidate in clf_results:
                if cls_result["is_fod"]:
                    det = {
                        "x1": candidate["x1"], "y1": candidate["y1"],
                        "x2": candidate["x2"], "y2": candidate["y2"],
                        "confidence": cls_result["confidence"],
                        "cnn_class":  cls_result["class_name"],
                    }
                    if cls_result["fast_path"]:
                        fast_cands.append(det)
                    else:
                        slow_cands.append(det)

            # Step 7 — Temporal confirmation
            # Slow path: must survive 2-frame prefilter before entering main tracker
            prefilter_confirmed = prefilter.update(slow_cands)
            # Fast path + prefilter graduates → main 3-frame tracker
            confirmed = tracker.update(fast_cands + prefilter_confirmed)

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
                    f"ALERT: {len(confirmed)} FOD confirmed — "
                    f"{avg_fps:.1f}fps — {frame_ms:.1f}ms"
                )
                if args.save:
                    vis = roi.copy()
                    for fod in confirmed:
                        cv2.rectangle(vis,
                                      (fod["x1"], fod["y1"]),
                                      (fod["x2"], fod["y2"]),
                                      (0, 0, 255), 2)
                        cv2.putText(vis, f"FOD {fod['confidence']:.2f}",
                                    (fod["x1"], max(0, fod["y1"] - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(str(alerts_dir / f"alert_{frame_count:05d}_{ts}.jpg"), vis)

            if args.visualise:
                vis = roi.copy()
                for fod in confirmed:
                    cv2.rectangle(vis,
                                  (fod["x1"], fod["y1"]),
                                  (fod["x2"], fod["y2"]),
                                  (0, 0, 255), 2)
                    cv2.putText(vis, f"FOD {fod['confidence']:.2f}",
                                (fod["x1"], max(0, fod["y1"] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                if confirmed:
                    cv2.putText(vis, f"ALERT: {len(confirmed)} FOD",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 3)
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
    logger.info("PRIME — Run complete")
    logger.info(f"  Frames    : {frame_count}")
    logger.info(f"  Alerts    : {alert_count}")
    logger.info(f"  Avg FPS   : {avg_fps_f:.1f}")
    logger.info(f"  Avg ms    : {1000/avg_fps_f:.1f}" if avg_fps_f > 0 else "")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
