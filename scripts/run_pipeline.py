"""
ARGUS-N — Main Pipeline Orchestrator (v2 — Multi-Pathway)

3-pathway architecture:
  A. PatchCore (DINOv2)  — anomaly detection, no training needed
  B. CLIP                — semantic FOD classification on crops
  C. RAFT optical flow   — motion residual, with bump detector

All 3 pathways run in parallel threads. Fusion gate combines results.
NIR gate applies dynamic threshold (mean + 2sigma) as a final filter.

Usage:
    cd /Volumes/T72/argusN
    PYTHONPATH=/Volumes/T72/argusN python scripts/run_pipeline.py
    PYTHONPATH=/Volumes/T72/argusN python scripts/run_pipeline.py --no-nir
    PYTHONPATH=/Volumes/T72/argusN python scripts/run_pipeline.py --build-bank
    PYTHONPATH=/Volumes/T72/argusN python scripts/run_pipeline.py --bank-frames 50

Flags:
    --build-bank        Rebuild PatchCore memory bank from current frame buffer
    --bank-frames N     Number of warmup frames to use for bank build (default 60)
    --no-nir            Disable NIR gate (useful when testing without NIR camera)
    --no-display        Suppress overlay window
    --device mps|cuda|cpu
"""

import argparse
import json
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

# -- Path setup ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force ALL model caches onto the drive — nothing touches the Mac system drive
import os
_CACHE = str(ROOT / "models" / "cache")
os.makedirs(_CACHE, exist_ok=True)
os.environ["TORCH_HOME"]          = _CACHE   # torch.hub + torch.load cache
os.environ["TRANSFORMERS_CACHE"]  = _CACHE   # HuggingFace (if used later)
os.environ["HF_HOME"]             = _CACHE
os.environ["XDG_CACHE_HOME"]      = _CACHE

from src.utils.config_loader import Config
from src.utils.logger import get_logger
from src.ingestion.multi_camera import MultiCameraIngestion
from src.ingestion.nir_simulator import NIRSimulator
from src.flow.raft_flow import RAFTFlow
from src.flow.residual import FlowResidual
from src.flow.egomotion import Egomotion
from src.flow.bump_detector import BumpDetector
from src.anomaly.patchcore import PatchCoreDetector
from src.semantic.clip_classifier import CLIPClassifier
from src.fusion.fusion import AdaptiveFusion
from ultralytics import YOLO as UltralyticsYOLO


# -- Overlay helpers -------------------------------------------------------
def _draw_alerts(frame: np.ndarray, alerts: list, meta: dict) -> np.ndarray:
    vis = frame.copy()
    for alert in alerts:
        x1, y1, x2, y2 = alert["box"]
        color = (0, 0, 255)  # red
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        label = f"FOD [{','.join(alert['pathways'])}] {alert['confidence']:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    info = (f"cand:{meta.get('candidates',0)}  "
            f"fast:{meta.get('fast_path',0)}  "
            f"temp:{meta.get('temporal_path',0)}  "
            f"nir_rej:{meta.get('nir_rejected',0)}  "
            f"bump:{meta.get('flow_discarded',False)}  "
            f"FOD:{len(alerts)}")
    cv2.putText(vis, info, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return vis


# -- Main ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ARGUS-N pipeline")
    parser.add_argument("--build-bank",   action="store_true",
                        help="Rebuild PatchCore memory bank from warmup frames")
    parser.add_argument("--bank-frames",  type=int, default=60,
                        help="Frames to collect for bank build (default 60)")
    parser.add_argument("--source",       default=None,
                        help="Path to video file (overrides config video_file_path)")
    parser.add_argument("--no-nir",       action="store_true",
                        help="Disable NIR gate")
    parser.add_argument("--no-display",   action="store_true",
                        help="Disable visualisation window")
    parser.add_argument("--device",       default=None,
                        help="Override device (mps|cuda|cpu)")
    parser.add_argument("--patchcore-stride", type=int, default=5,
                        help="Run PatchCore every N frames (default 5).")
    parser.add_argument("--clip-stride",  type=int, default=10,
                        help="Run CLIP independently every N frames (default 10). "
                             "Catches FOD even when PatchCore and flow miss.")
    parser.add_argument("--floor-crop",   type=float, default=0.35,
                        help="Fraction of frame TOP to discard as non-floor "
                             "(default 0.35 = top 35%%). Cuts out sky/walls/machinery.")
    parser.add_argument("--pc-threshold", type=float, default=0.35,
                        help="PatchCore anomaly threshold (default 0.35, lower = more sensitive)")
    args = parser.parse_args()

    # -- Config & logging --------------------------------------------------
    cfg_path = ROOT / "config" / "config.yaml"
    cfg      = Config(str(cfg_path))
    log      = get_logger(
        "pipeline",
        cfg.get("logging", "log_path", default="logs/argus.log"),
        cfg.get("logging", "level", default="INFO"),
    )

    device = args.device or cfg.get("device", default="mps")
    log.info(f"ARGUS-N v2 starting — device={device}")

    # -- Output dirs -------------------------------------------------------
    out_det   = ROOT / cfg.get("outputs", "detections_path",   default="outputs/detections")
    out_anom  = ROOT / cfg.get("outputs", "anomaly_frames_path", default="outputs/anomaly_frames")
    out_gps   = ROOT / cfg.get("outputs", "gps_logs_path",     default="outputs/gps_logs")
    for d in [out_det, out_anom, out_gps]:
        d.mkdir(parents=True, exist_ok=True)

    # -- Initialise components --------------------------------------------
    log.info("Initialising camera ingestion ...")
    camera = MultiCameraIngestion(cfg)
    if args.source:
        log.info(f"Overriding video source: {args.source}")
        camera.set_source(args.source)

    log.info("Initialising NIR simulator ...")
    nir_sim = NIRSimulator()

    log.info("Initialising optical flow ...")
    raft      = RAFTFlow(cfg)
    residual  = FlowResidual(cfg)
    egomotion = Egomotion(cfg)
    bump_det  = BumpDetector(window=30, k=3.0)

    log.info("Initialising PatchCore (Pathway A) ...")
    bank_path = ROOT / "models" / "patchcore_bank.pt"
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    patchcore = PatchCoreDetector(device=device, bank_path=str(bank_path))

    log.info("Initialising CLIP (Pathway B) ...")
    clip_clf = CLIPClassifier(device=device)

    log.info("Initialising YOLO detector (Pathway D) ...")
    _yolo_path = str(ROOT / cfg.get("yolo", "model_path",
                     default="models/yolo_runs/fod_v2/weights/best.pt"))
    try:
        yolo_d = UltralyticsYOLO(_yolo_path)
        log.info(f"YOLO Pathway D: {_yolo_path}")
    except Exception as _e:
        yolo_d = None
        log.warning(f"YOLO Pathway D not loaded: {_e}")

    log.info("Initialising fusion gate ...")
    nir_enabled = cfg.get("fusion", "nir_gate_enabled", default=True) and not args.no_nir
    fusion = AdaptiveFusion(
        patchcore_threshold = 0.5,
        clip_threshold      = 0.5,
        iou_min             = 0.2,
        temporal_frames     = 3,
        nir_enabled         = nir_enabled,
        nir_initial_thresh  = cfg.get("fusion", "nir_contrast_threshold", default=18.0),
    )

    # -- Warmup: collect clean frames for PatchCore bank ------------------
    warmup_n    = cfg.get("pipeline", "warmup_frames", default=60)
    bank_frames = args.bank_frames
    warm_buffer = []

    log.info(f"Warming up — collecting {warmup_n} frames ...")

    if args.no_display:
        window_open = False
    else:
        cv2.namedWindow("ARGUS-N", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ARGUS-N", 1280, 720)
        window_open = True

    frame_idx   = 0
    alert_log   = []
    floor_y0    = 0   # top pixel of floor ROI — set after first frame

    # CLIP standalone stride counter
    clip_stride_counter = 0

    for frames_rgb, frame_nir in camera:
        if frames_rgb is None or len(frames_rgb) == 0:
            continue

        primary = frames_rgb[0]   # primary RGB camera (camera 0)

        # -- Floor ROI crop -----------------------------------------------
        # Discard top N% of frame (sky / walls / machinery) so PatchCore and
        # CLIP only see the floor surface.  Re-map alert boxes back to full
        # frame coordinates afterwards.
        if floor_y0 == 0 and args.floor_crop > 0:
            floor_y0 = int(primary.shape[0] * args.floor_crop)
        floor = primary[floor_y0:, :]   # crop: floor only

        # Simulate NIR if not real
        if frame_nir is None and cfg.get("camera", "simulate_nir", default=False):
            frame_nir = nir_sim.simulate(floor)
        elif frame_nir is not None:
            frame_nir = frame_nir[floor_y0:, :]

        # Collect warmup frames (floor crop only — clean surface features)
        if frame_idx < warmup_n:
            warm_buffer.append(floor.copy())
            frame_idx += 1
            if window_open:
                vis = primary.copy()
                cv2.putText(vis, f"Warmup {frame_idx}/{warmup_n}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (200, 200, 0), 2)
                cv2.imshow("ARGUS-N", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # -- Build PatchCore bank once warmup is done --------------------
        if frame_idx == warmup_n:
            if args.build_bank or patchcore.bank is None:
                log.info(f"Building PatchCore memory bank from "
                         f"{min(bank_frames, len(warm_buffer))} frames ...")
                build_frames = warm_buffer[:bank_frames]
                patchcore.build_memory_bank(build_frames)
            else:
                log.info("PatchCore memory bank loaded from disk — skipping build.")
            frame_idx += 1  # prevent re-entry

        # == DETECTION PHASE =============================================
        # Compute-gated pipeline:
        #   1. Flow runs EVERY frame  (fast — Farneback CPU)
        #   2. PatchCore runs every --patchcore-stride frames OR when flow
        #      finds candidates (whichever comes first)
        #   3. CLIP runs ONLY when PatchCore or flow returns boxes

        # -- Pathway C: flow (every frame, fast) -------------------------
        flow_t0 = time.perf_counter()
        flow = raft.compute(floor)
        flow_boxes    = []
        flow_discarded = False

        if flow is not None:
            flow_discarded = bump_det.update(flow)
            if not flow_discarded:
                expected_flow = egomotion.compute_expected_flow()
                if expected_flow is not None and expected_flow.shape != flow.shape:
                    expected_flow = cv2.resize(
                        expected_flow,
                        (flow.shape[1], flow.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                if expected_flow is None:
                    expected_flow = np.zeros_like(flow)
                _, _, candidates = residual.compute(flow, expected_flow)
                flow_boxes = [
                    (c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"])
                    for c in candidates
                ]
        flow_ms = (time.perf_counter() - flow_t0) * 1000

        # -- Pathway A: PatchCore (gated) --------------------------------
        detection_frame = frame_idx - warmup_n
        run_pc = (
            len(flow_boxes) > 0
            or (detection_frame % args.patchcore_stride == 0)
        )

        pa_result = {"score": 0.0, "boxes": [], "ms": 0.0, "ran": False}
        pb_result = {"results": [], "ms": 0.0}

        if run_pc:
            t0 = time.perf_counter()
            score, _ = patchcore.score(floor, return_heatmap=False)
            boxes     = patchcore.get_candidate_regions(
                            floor, threshold=args.pc_threshold)
            pa_result = {"score": score, "boxes": boxes,
                         "ms": (time.perf_counter() - t0) * 1000, "ran": True}

        # -- Pathway B: CLIP ---------------------------------------------
        # Runs when any pathway has boxes, OR independently every clip_stride frames
        clip_stride_counter += 1
        run_clip_independent = (clip_stride_counter % args.clip_stride == 0)
        candidate_boxes = pa_result["boxes"] + flow_boxes

        if candidate_boxes or run_clip_independent:
            t0 = time.perf_counter()
            if run_clip_independent and not candidate_boxes:
                # Full-frame scan: divide floor into a 3×2 grid and score each tile
                fh, fw = floor.shape[:2]
                tile_boxes = []
                for row in range(2):
                    for col in range(3):
                        tx1 = col * fw // 3
                        tx2 = (col + 1) * fw // 3
                        ty1 = row * fh // 2
                        ty2 = (row + 1) * fh // 2
                        tile_boxes.append((tx1, ty1, tx2, ty2))
                clip_results_raw = clip_clf.score_regions(floor, tile_boxes)
            else:
                clip_results_raw = clip_clf.score_regions(floor, candidate_boxes)
            pb_result["results"] = clip_results_raw
            pb_result["ms"]      = (time.perf_counter() - t0) * 1000

        # -- Pathway D: YOLO (every frame, ~3ms, supervised) ---------------
        pd_result = {"detections": [], "ms": 0.0}
        if yolo_d is not None:
            _t0 = time.perf_counter()
            _dev = "cuda" if "cuda" in device else device
            _raw = yolo_d.predict(floor, conf=0.25, verbose=False, device=_dev)
            if _raw and _raw[0].boxes is not None:
                for _b in _raw[0].boxes:
                    _x1,_y1,_x2,_y2 = map(int, _b.xyxy[0].tolist())
                    pd_result["detections"].append({
                        "box":      (_x1, _y1, _x2, _y2),
                        "conf":     float(_b.conf[0]),
                        "cls_name": yolo_d.names[int(_b.cls[0])],
                    })
            pd_result["ms"] = (time.perf_counter() - _t0) * 1000

        # -- Remap floor-crop boxes back to full-frame coords ------------
        def remap(boxes):
            return [(x1, y1 + floor_y0, x2, y2 + floor_y0) for x1,y1,x2,y2 in boxes]
        def remap_results(results):
            return [{**r, "box": (r["box"][0], r["box"][1] + floor_y0,
                                   r["box"][2], r["box"][3] + floor_y0)}
                    for r in results]

        pa_boxes_full  = remap(pa_result["boxes"])
        flow_boxes_full = remap(flow_boxes)
        clip_results_full = remap_results(pb_result["results"])
        yolo_dets_full    = [
            {**d, "box": (d["box"][0], d["box"][1] + floor_y0,
                            d["box"][2], d["box"][3] + floor_y0)}
            for d in pd_result["detections"]
        ]

        # -- Fusion (uses full-frame coords for NIR gate) -----------------
        alerts, meta = fusion.fuse(
            frame_bgr       = primary,
            frame_nir       = frame_nir,
            patchcore_boxes = pa_boxes_full,
            patchcore_score = pa_result.get("score", 0.0),
            clip_results    = clip_results_full,
            flow_boxes      = flow_boxes_full,
            flow_discarded  = flow_discarded,
            yolo_detections = yolo_dets_full,
        )

        meta["pa_ms"]   = pa_result.get("ms", 0)
        meta["pb_ms"]   = pb_result.get("ms", 0)
        meta["pd_ms"]   = pd_result.get("ms", 0)
        meta["flow_ms"] = flow_ms

        # -- Logging & persistence ----------------------------------------
        if alerts:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            for alert in alerts:
                log.warning(
                    f"FOD ALERT  box={alert['box']}  "
                    f"pathways={alert['pathways']}  "
                    f"conf={alert['confidence']:.3f}  "
                    f"fast={alert['fast_path']}  "
                    f"nir_contrast={alert['nir_contrast']:.1f}"
                )
            # Save annotated frame
            if cfg.get("outputs", "save_anomaly_frames", default=True):
                frame_path = out_anom / f"fod_{ts}.jpg"
                save_vis   = _draw_alerts(primary, alerts, meta)
                cv2.imwrite(str(frame_path), save_vis)

            # Append to JSON alert log
            for alert in alerts:
                alert_log.append({
                    "timestamp": ts,
                    "frame":     frame_idx,
                    "box":       alert["box"],
                    "pathways":  alert["pathways"],
                    "confidence": alert["confidence"],
                    "nir_contrast": alert["nir_contrast"],
                    "fast_path":  alert["fast_path"],
                })
        else:
            log.debug(
                f"frame={frame_idx}  "
                f"cand={meta['candidates']}  "
                f"bump={meta['flow_discarded']}  "
                f"pa={pa_result.get('score',0):.3f}  "
                f"FOD=0"
            )

        # -- Visualisation ------------------------------------------------
        if window_open:
            vis = _draw_alerts(primary, alerts, meta)
            cv2.imshow("ARGUS-N", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                log.info("User quit.")
                break
        else:
            # Headless: brief sleep so we don't pin the CPU/GPU at 100%
            # At 30km/h the vehicle moves ~8mm per ms — 30ms per frame is fine
            time.sleep(0.03)

        frame_idx += 1

    # -- Teardown ----------------------------------------------------------
    camera.release()
    if window_open:
        cv2.destroyAllWindows()

    # Save alert log
    log_path = out_det / f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(alert_log, indent=2))
    log.info(f"Pipeline complete.  Total alerts: {len(alert_log)}  Log: {log_path}")


if __name__ == "__main__":
    main()
