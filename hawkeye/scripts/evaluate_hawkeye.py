"""
evaluate_hawkeye.py — Evaluate HAWKEYE on the held-out test set.

Computes all ARGUS-N standard metrics (identical schema to yolofinetune):
  - mAP50, mAP50-95, Precision, Recall, F1  (via YOLO val on test split)
  - False Positive Rate  (full HAWKEYE pipeline on clean tarmac video)
  - Inference FPS and latency ms/frame  (full HAWKEYE pipeline timing)

Results saved to logs/eval_results.json.

Usage (run from inside hawkeye/ directory):
    # Full evaluation:
    python scripts/evaluate_hawkeye.py \\
        --data config/dataset.yaml \\
        --clean-video ../yolofinetune/data/raw/videos/clean_runway.mp4

    # Detection metrics only:
    python scripts/evaluate_hawkeye.py --data config/dataset.yaml

    # FP rate only:
    python scripts/evaluate_hawkeye.py \\
        --clean-video ../yolofinetune/data/raw/videos/clean_runway.mp4
"""

import cv2
import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.flow.farneback import FarnebackFlow
from src.flow.egomotion import Egomotion
from src.flow.residual import FlowResidual
from src.detection.yolo_detector import YOLODetector
from src.anomaly.patchcore import PatchCore
from src.fusion.hawkeye_fusion import HawkeyeFusion


def apply_roi_crop(frame, top_frac: float, bot_frac: float):
    h, w = frame.shape[:2]
    y_start = int(h * top_frac)
    y_end   = int(h * (1.0 - bot_frac))
    return frame[y_start:y_end, :], y_start


def evaluate_on_dataset(model, data_yaml: Path, imgsz: int, device: str, log) -> dict:
    """
    Run YOLO val() on test split for mAP50, mAP50-95, precision, recall.
    Uses the YOLO component directly — consistent with YOLOFINETUNE baseline.
    """
    log.info(f"Running YOLO val() on {data_yaml} (test split)...")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    p  = float(metrics.box.mp)
    r  = float(metrics.box.mr)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    results = {
        "mAP50":     float(metrics.box.map50),
        "mAP50_95":  float(metrics.box.map),
        "precision": p,
        "recall":    r,
        "f1":        f1,
    }

    log.info(f"mAP50     : {results['mAP50']:.4f}")
    log.info(f"mAP50-95  : {results['mAP50_95']:.4f}")
    log.info(f"Precision : {results['precision']:.4f}")
    log.info(f"Recall    : {results['recall']:.4f}")
    log.info(f"F1        : {results['f1']:.4f}")
    return results


def benchmark_pipeline_fps(
    cfg, yolo_model, patchcore, log, n_frames: int = 200
) -> dict:
    """
    Time the FULL HAWKEYE pipeline (all three components + fusion) on blank frames.
    Returns latency_ms and fps for end-to-end throughput.
    """
    import numpy as np

    log.info(f"Pipeline FPS benchmark ({n_frames} synthetic frames)...")

    farneback = FarnebackFlow(cfg)
    egomotion = Egomotion(cfg)
    residual  = FlowResidual(cfg)
    fusion    = HawkeyeFusion(cfg)

    top_crop  = cfg.get("pipeline", "top_crop",  default=0.22)
    bot_crop  = cfg.get("pipeline", "bot_crop",  default=0.15)
    imgsz     = cfg.get("yolo", "input_size",    default=640)
    device    = cfg.device

    # Synthetic 1080p frame
    H, W = 1080, 1920
    ch   = int(H * (1.0 - top_crop - bot_crop))

    # Prime the flow module (needs two frames)
    dummy = np.zeros((ch, W, 3), dtype=np.uint8)
    farneback.compute(dummy)

    latencies = []
    for _ in range(n_frames):
        frame = np.random.randint(0, 255, (ch, W, 3), dtype=np.uint8)
        t0 = time.perf_counter()

        yolo_dets = yolo_model.predict(
            frame, imgsz=imgsz, conf=cfg.get("yolo","confidence_threshold",default=0.35),
            iou=cfg.get("yolo","iou_threshold",default=0.45), verbose=False, device=device
        )
        flow = farneback.compute(frame)
        if flow is not None:
            exp  = egomotion.compute_expected_flow(ch, W)
            _, _, cands = residual.compute(flow, exp)
            _ = fusion.fuse(frame, [], cands, patchcore)

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    fps    = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    log.info(f"Latency   : {avg_ms:.1f} ms/frame")
    log.info(f"FPS       : {fps:.1f}")
    return {"latency_ms": avg_ms, "fps": fps}


def evaluate_false_positive_rate(cfg, yolo_model, patchcore, video_path: str, log) -> dict:
    """
    Run the full HAWKEYE fusion pipeline on a clean tarmac video (zero FODs).
    Every alert raised is a false positive.
    Returns fp_per_minute, fp_total, clean_video_duration_s.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        log.warning(f"Clean video not found: {video_path}")
        return {"fp_per_minute": None, "fp_total": None, "clean_video_duration_s": None}

    top_crop = cfg.get("pipeline", "top_crop", default=0.22)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.15)
    imgsz    = cfg.get("yolo", "input_size", default=640)
    conf     = cfg.get("yolo", "confidence_threshold", default=0.35)
    iou      = cfg.get("yolo", "iou_threshold", default=0.45)
    device   = cfg.device

    farneback = FarnebackFlow(cfg)
    egomotion = Egomotion(cfg)
    residual  = FlowResidual(cfg)
    fusion    = HawkeyeFusion(cfg)

    cap = cv2.VideoCapture(str(video_path))
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / source_fps

    log.info(f"Clean video : {video_path.name} — {total_frames} frames ({duration_s:.1f}s)")

    fp_count  = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        h, w = frame.shape[:2]
        y_start = int(h * top_crop)
        y_end   = int(h * (1.0 - bot_crop))
        cropped = frame[y_start:y_end, :]
        ch, cw  = cropped.shape[:2]

        # YOLO component
        results   = yolo_model.predict(cropped, imgsz=imgsz, conf=conf,
                                       iou=iou, verbose=False, device=device)
        yolo_dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                yolo_dets.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "confidence": float(box.conf[0]),
                    "class_id": 0, "class_name": "fod"
                })

        # Flow component
        flow = farneback.compute(cropped)
        cands = []
        if flow is not None:
            exp_flow = egomotion.compute_expected_flow(ch, cw)
            _, _, cands = residual.compute(flow, exp_flow)

        # Fusion
        alerts = fusion.fuse(cropped, yolo_dets, cands, patchcore)
        if alerts:
            fp_count += 1  # one false alert per frame

    cap.release()

    fp_per_min = (fp_count / duration_s * 60) if duration_s > 0 else 0.0
    log.info(f"FP total    : {fp_count} alerts on {frame_idx} clean frames")
    log.info(f"FP rate     : {fp_per_min:.2f} per minute")

    return {
        "fp_per_minute":          fp_per_min,
        "fp_total":               fp_count,
        "clean_video_duration_s": duration_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate HAWKEYE on test set.")
    parser.add_argument("--model",       default="models/yolo/finetuned/best.pt")
    parser.add_argument("--data",        default="config/dataset.yaml", help="Dataset YAML")
    parser.add_argument("--clean-video", default=None, help="Clean runway video for FP rate")
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--device",      default=None)
    parser.add_argument("--output",      default="logs/eval_results.json")
    args = parser.parse_args()

    from ultralytics import YOLO

    cfg = load_config(args.config)
    log = get_logger(
        "evaluate_hawkeye",
        cfg.get("logging", "log_path", default="logs/hawkeye.log"),
        cfg.get("logging", "level",    default="INFO"),
    )

    if args.device:
        cfg._cfg["device"] = args.device

    device = cfg.device
    imgsz  = cfg.get("yolo", "input_size", default=640)

    model_path = Path(args.model)
    if not model_path.exists():
        log.error(f"YOLO weights not found: {model_path}")
        log.error("Copy from: cp ../yolofinetune/models/yolo/finetuned/best.pt models/yolo/finetuned/best.pt")
        sys.exit(1)

    log.info("=" * 55)
    log.info("HAWKEYE — Evaluation")
    log.info("=" * 55)
    log.info(f"Model  : {model_path}")
    log.info(f"Device : {device}")

    yolo_model = YOLO(str(model_path))

    log.info("Loading PatchCore bank...")
    patchcore = PatchCore(cfg)
    if patchcore.memory_bank is None:
        log.warning("PatchCore bank not loaded — FP rate will be under-counted.")
        log.warning("Run build_patchcore_bank.py first.")

    results = {"model": "hawkeye", "model_path": str(model_path)}

    # ── Detection metrics (YOLO val on test split) ─────────────────────────
    data_yaml = Path(args.data)
    if data_yaml.exists():
        det = evaluate_on_dataset(yolo_model, data_yaml, imgsz, device, log)
        results.update(det)
    else:
        log.warning(f"Dataset YAML not found: {data_yaml} — skipping mAP metrics")

    # ── Full pipeline FPS benchmark ────────────────────────────────────────
    log.info("\nBenchmarking full pipeline speed...")
    speed = benchmark_pipeline_fps(cfg, yolo_model, patchcore, log)
    results.update(speed)

    # ── False positive rate on clean video ─────────────────────────────────
    if args.clean_video:
        log.info("\nMeasuring false positive rate...")
        fp = evaluate_false_positive_rate(
            cfg, yolo_model, patchcore, args.clean_video, log
        )
        results.update(fp)

    # ── Save ───────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nResults saved → {out}")
    log.info("\n── Summary ──")
    for k, v in results.items():
        if k not in ("model", "model_path"):
            log.info(f"  {k:<30}: {v}")


if __name__ == "__main__":
    main()
