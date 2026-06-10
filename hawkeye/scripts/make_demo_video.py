"""
make_demo_video.py — Run HAWKEYE on FOD footage and write annotated output video.

Produces outputs/demo_hawkeye_fod1.mp4 — visually comparable to
yolofinetune/outputs/demo_yolofinetune_fod1.mp4 for side-by-side comparison.

Overlay style matches YOLOFINETUNE:
  - Red bounding boxes around each confirmed FOD
  - "ALERT: N FOD DETECTED" banner at top when alert raised
  - Vote breakdown label per box: "FOD [2/3] Y=1 F=1 P=0"
  - FPS counter top-right
  - Model name watermark bottom-left

Usage (run from inside hawkeye/ directory):
    python scripts/make_demo_video.py \\
        --video ../yolofinetune/data/raw/videos/fod_sessions/fod1.mp4 \\
        --output outputs/demo_hawkeye_fod1.mp4

    # Preview while rendering:
    python scripts/make_demo_video.py \\
        --video ../yolofinetune/data/raw/videos/fod_sessions/fod1.mp4 \\
        --output outputs/demo_hawkeye_fod1.mp4 \\
        --preview
"""

import cv2
import sys
import time
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


def draw_alerts(frame, alerts: list, y_offset: int, alert_count: int):
    """
    Draw confirmed HAWKEYE alerts onto the full (un-cropped) frame.
    Matches YOLOFINETUNE overlay style: red boxes, alert banner.
    """
    for alert in alerts:
        x1 = alert["x"]
        y1 = alert["y"] + y_offset
        x2 = x1 + alert["w"]
        y2 = y1 + alert["h"]

        # Red bounding box (same as yolofinetune)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Vote breakdown label
        label = (
            f"FOD [{alert['votes']}/3] "
            f"Y={alert['yolo_vote']} "
            f"F={alert['flow_vote']} "
            f"P={alert['patchcore_vote']}"
        )
        cv2.putText(
            frame, label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
        )

    if alerts:
        cv2.putText(
            frame, f"ALERT: {len(alerts)} FOD DETECTED",
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
        )

    return frame


def main():
    parser = argparse.ArgumentParser(description="Render HAWKEYE demo video.")
    parser.add_argument(
        "--video", required=True,
        help="Input video path (fod1.mp4)"
    )
    parser.add_argument(
        "--output", default="outputs/demo_hawkeye_fod1.mp4",
        help="Output video path (default: outputs/demo_hawkeye_fod1.mp4)"
    )
    parser.add_argument(
        "--config", default="config/config.yaml"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show live preview window while rendering"
    )
    parser.add_argument(
        "--speed", type=float, default=None,
        help="Simulated vehicle speed km/h (overrides config)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = get_logger(
        "make_demo_video",
        cfg.get("logging", "log_path", default="logs/hawkeye.log"),
        cfg.get("logging", "level",    default="INFO"),
    )

    video_path = Path(args.video)
    if not video_path.exists():
        log.error(f"Video not found: {video_path}")
        sys.exit(1)

    if args.speed:
        cfg._cfg["imu"]["simulated_speed_kmh"] = args.speed

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_crop    = cfg.get("pipeline", "top_crop",    default=0.22)
    bot_crop    = cfg.get("pipeline", "bot_crop",    default=0.15)
    warmup      = cfg.get("pipeline", "warmup_frames", default=30)
    imgsz       = cfg.get("yolo",     "input_size",  default=640)
    conf        = cfg.get("yolo",     "confidence_threshold", default=0.35)
    iou         = cfg.get("yolo",     "iou_threshold",        default=0.45)
    device      = cfg.device

    # ── Probe source video ─────────────────────────────────────────────────
    cap_probe = cv2.VideoCapture(str(video_path))
    source_fps   = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    log.info("=" * 55)
    log.info("HAWKEYE — Demo Video")
    log.info("=" * 55)
    log.info(f"Input  : {video_path.name}  ({orig_w}x{orig_h} @ {source_fps:.0f}fps, {total_frames} frames)")
    log.info(f"Output : {output_path}")

    # ── VideoWriter ────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, source_fps, (orig_w, orig_h))
    if not writer.isOpened():
        log.error("VideoWriter failed to open — check output path and codec")
        sys.exit(1)

    # ── Load pipeline components ───────────────────────────────────────────
    from ultralytics import YOLO
    log.info("Loading pipeline components...")
    yolo_model = YOLO(cfg.get("yolo", "model_path", default="models/yolo/finetuned/best.pt"))
    farneback  = FarnebackFlow(cfg)
    egomotion  = Egomotion(cfg)
    residual   = FlowResidual(cfg)
    patchcore  = PatchCore(cfg)
    fusion     = HawkeyeFusion(cfg)

    if patchcore.memory_bank is None:
        log.warning("PatchCore bank not loaded — PatchCore vote will always be 0")

    # ── Open video and burn warmup frames ──────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    for _ in range(warmup):
        cap.read()

    # ── Main render loop ───────────────────────────────────────────────────
    frame_idx   = 0
    alert_count = 0
    fps_timer   = time.time()
    fps_display = 0.0

    log.info("Rendering...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.time() - fps_timer
                fps_display = 30.0 / elapsed if elapsed > 0 else 0.0
                fps_timer = time.time()

            cropped, y_offset = apply_roi_crop(frame, top_crop, bot_crop)
            ch, cw = cropped.shape[:2]

            # YOLO
            results = yolo_model.predict(
                cropped, imgsz=imgsz, conf=conf, iou=iou,
                verbose=False, device=device
            )
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

            # Flow
            flow  = farneback.compute(cropped)
            cands = []
            if flow is not None:
                exp_flow = egomotion.compute_expected_flow(ch, cw)
                _, _, cands = residual.compute(flow, exp_flow)

            # Fusion
            alerts = fusion.fuse(cropped, yolo_dets, cands, patchcore)
            if alerts:
                alert_count += 1

            # ── Draw onto full frame ──────────────────────────────────────
            out_frame = frame.copy()
            out_frame = draw_alerts(out_frame, alerts, y_offset, alert_count)

            # FPS counter (top-right, matches yolofinetune style)
            cv2.putText(
                out_frame, f"FPS: {fps_display:.1f}",
                (orig_w - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            # Model watermark (bottom-left)
            cv2.putText(
                out_frame, "HAWKEYE",
                (20, orig_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2
            )

            writer.write(out_frame)

            if args.preview:
                cv2.imshow("HAWKEYE — Demo", out_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("User quit preview")
                    break

            if frame_idx % 100 == 0:
                log.info(f"  Frame {frame_idx}/{total_frames} — alerts so far: {alert_count}")

    finally:
        cap.release()
        writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    log.info("=" * 55)
    log.info(f"Done — {frame_idx} frames rendered")
    log.info(f"Alert frames : {alert_count}")
    log.info(f"Output       : {output_path}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
