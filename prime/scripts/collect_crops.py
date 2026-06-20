"""
PRIME — collect_crops.py
Run YOLO on footage and save every detection as a 4-channel crop
(BGR locally-normalised + frame-diff) to data/crops/raw_crops/.

4th channel: absolute frame difference (current − previous, grayscale).
Physics: stationary FOD → near-zero diff.  Moving artefacts → non-zero.

Local contrast normalisation is applied inside CropBuilder.build() —
the CNN sees texture anomaly relative to local background, not absolute
runway colour.  This generalises across different runway surfaces.

IMPORTANT — clean1.mp4 hard split:
    Only frames 0..CLEAN_SPLIT_FRAME-1 may be used for crop collection.
    Frames CLEAN_SPLIT_FRAME+ are held out strictly for FP evaluation.
    Pass --max-frame 6300 when running on the clean video.

Usage (from inside prime/):
    # FOD footage — no frame limit
    python scripts/collect_crops.py \\
        --source ../yolofinetune/data/raw/videos/fod_sessions/fod1.mp4

    # Clean footage — HARD LIMIT at frame 6300
    python scripts/collect_crops.py \\
        --source ../yolofinetune/data/raw/videos/clean_runway.mp4 \\
        --max-frame 6300
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.flow.frame_diff import FrameDiff
from src.semantic.crop_builder import CropBuilder


# ── Geometric pre-filter constants ──────────────────────────────────────────
# Applied before any ML — kills obvious non-FOD without model inference.
MAX_ASPECT_RATIO = 4.0    # FOD is compact; wider box → likely marking or shadow
MAX_FRAME_AREA_FRAC = 0.05  # FOD occupies < 5% of frame area


def apply_roi(frame, top_crop, bot_crop):
    h = frame.shape[0]
    y0 = int(h * top_crop)
    y1 = int(h * (1.0 - bot_crop))
    return frame[y0:y1, :], y0


def passes_geometric_filter(det: dict, frame_h: int, frame_w: int) -> bool:
    """
    Return False for boxes that are obviously not FOD based on shape alone.
    No model, no calibration — pure dimensionless ratios.
    """
    w = max(1, det["x2"] - det["x1"])
    h = max(1, det["y2"] - det["y1"])

    aspect = max(w / h, h / w)
    if aspect > MAX_ASPECT_RATIO:
        return False

    area_frac = (w * h) / max(1, frame_h * frame_w)
    if area_frac > MAX_FRAME_AREA_FRAC:
        return False

    return True


def collect_from_video(video_path, output_dir, yolo_model, frame_diff,
                       crop_builder, cfg, logger, max_frame: int = 0):
    top_crop = cfg.get("pipeline", "top_crop", default=0.50)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.05)
    imgsz    = cfg.get("yolo", "input_size",           default=640)
    conf_t   = cfg.get("yolo", "confidence_threshold", default=0.28)
    iou_t    = cfg.get("yolo", "iou_threshold",        default=0.45)
    device   = cfg.device

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frame > 0:
        total = min(total, max_frame)

    stem  = video_path.stem
    saved = 0
    frame_diff.reset()

    for frame_idx in tqdm(range(total), desc=f"  {stem}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        roi, _ = apply_roi(frame, top_crop, bot_crop)
        roi_h, roi_w = roi.shape[:2]

        # Frame-diff 4th channel — computed before YOLO to keep timing consistent
        diff_mag = frame_diff.compute(roi)   # (H, W) float32 [0,255]

        # YOLO — sole candidate source
        results = yolo_model.predict(roi, imgsz=imgsz, conf=conf_t,
                                     iou=iou_t, verbose=False, device=device)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                d = {
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "confidence": float(box.conf[0]),
                }
                if passes_geometric_filter(d, roi_h, roi_w):
                    dets.append(d)

        if not dets:
            continue

        # Build and save 4-channel crops
        for ci, (crop, candidate) in enumerate(
            crop_builder.build_batch(roi, diff_mag, dets)
        ):
            if crop is None:
                continue
            fname = f"{stem}_f{frame_idx:06d}_c{ci:03d}.png"
            crop_builder.save_crop(crop, str(output_dir / fname))
            saved += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Collect 4-channel CNN training crops from video footage"
    )
    parser.add_argument("--source",    required=True,
                        help="Video file or directory of video files")
    parser.add_argument("--output",    default="data/crops/raw_crops")
    parser.add_argument("--config",    default="config/config.yaml")
    parser.add_argument("--ext",       default=".mp4,.avi,.mov,.MP4")
    parser.add_argument("--max-frame", type=int, default=0,
                        help="Stop after this many frames (0 = no limit). "
                             "Pass 6300 for clean1.mp4 to respect the hard split.")
    args = parser.parse_args()

    from ultralytics import YOLO

    cfg    = load_config(args.config)
    logger = get_logger("collect_crops",
                        cfg.get("logging", "log_path", default="logs/prime.log"),
                        cfg.get("logging", "level",    default="INFO"))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = set(args.ext.split(","))
    source     = Path(args.source)
    video_files = ([source] if source.is_file()
                   else [f for f in source.rglob("*") if f.suffix in extensions])

    if not video_files:
        logger.error(f"No video files found at {source}")
        sys.exit(1)

    logger.info(f"Found {len(video_files)} video(s) — output → {output_dir}")
    if args.max_frame > 0:
        logger.info(f"Frame limit : {args.max_frame} (clean split boundary)")

    model_path   = cfg.get("yolo", "model_path", default="models/yolo/finetuned/best.pt")
    yolo_model   = YOLO(model_path)
    frame_diff   = FrameDiff()
    crop_builder = CropBuilder(cfg)

    total_saved = 0
    for vf in video_files:
        logger.info(f"Processing {vf.name}")
        n = collect_from_video(vf, output_dir, yolo_model, frame_diff,
                               crop_builder, cfg, logger,
                               max_frame=args.max_frame)
        logger.info(f"  → {n} crops saved")
        total_saved += n

    logger.info(f"Done — {total_saved} total crops → {output_dir}")
    logger.info(f"Next: send raw_crops/ to teammate for annotation")
    logger.info(f"      python scripts/label_crops_standalone.py "
                f"--input {output_dir} --output data/crops/labeled_crops")


if __name__ == "__main__":
    main()
