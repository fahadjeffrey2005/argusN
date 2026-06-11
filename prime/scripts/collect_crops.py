"""
PRIME — collect_crops.py
Run YOLO on all available footage, save every detection as a 4-channel crop
(BGR + Farneback flow magnitude) to data/crops/raw_crops/.

NOTE: On static footage, flow magnitude will be near-zero.
The CNN still receives a valid 4-channel tensor — near-zero flow is itself
a signal (no physics anomaly). Collect from all footage regardless.

YOLO is the sole candidate source. Flow is computed only for the 4th channel.

Usage (from inside prime/):
    python scripts/collect_crops.py \\
        --source ../yolofinetune/data/raw/videos/fod_sessions/fod1.mp4

    python scripts/collect_crops.py \\
        --source ../yolofinetune/data/raw/videos/clean_runway.mp4

    python scripts/collect_crops.py \\
        --source ../yolofinetune/data/raw/videos/
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
from src.flow.farneback import FarnebackFlow
from src.semantic.crop_builder import CropBuilder


def apply_roi(frame, top_crop, bot_crop):
    h = frame.shape[0]
    y0 = int(h * top_crop)
    y1 = int(h * (1.0 - bot_crop))
    return frame[y0:y1, :], y0


def collect_from_video(video_path, output_dir, yolo_model, farneback,
                       crop_builder, cfg, logger):
    from ultralytics import YOLO

    top_crop = cfg.get("pipeline", "top_crop", default=0.50)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.05)
    imgsz    = cfg.get("yolo", "input_size",   default=640)
    conf_t   = cfg.get("yolo", "confidence_threshold", default=0.28)
    iou_t    = cfg.get("yolo", "iou_threshold",         default=0.45)
    device   = cfg.device

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stem  = video_path.stem
    saved = 0
    farneback.reset()

    for frame_idx in tqdm(range(total), desc=f"  {stem}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        roi, _ = apply_roi(frame, top_crop, bot_crop)
        roi_h, roi_w = roi.shape[:2]

        # YOLO detection — sole candidate source
        results = yolo_model.predict(roi, imgsz=imgsz, conf=conf_t,
                                     iou=iou_t, verbose=False, device=device)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "confidence": float(box.conf[0]),
                })

        if not dets:
            farneback.compute(roi)  # keep flow state updated even on empty frames
            continue

        # Flow magnitude for CNN 4th channel (may be ~0 on static footage)
        flow     = farneback.compute(roi)
        flow_mag = (farneback.magnitude_map(flow) if flow is not None
                    else np.zeros((roi_h, roi_w), dtype=np.float32))

        # Build and save 4-channel crops
        for ci, (crop, candidate) in enumerate(
            crop_builder.build_batch(roi, flow_mag, dets)
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
    parser.add_argument("--source", required=True,
                        help="Video file or directory of video files")
    parser.add_argument("--output", default="data/crops/raw_crops")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--ext",    default=".mp4,.avi,.mov,.MP4")
    args = parser.parse_args()

    from ultralytics import YOLO

    cfg    = load_config(args.config)
    logger = get_logger("collect_crops",
                        cfg.get("logging", "log_path", default="logs/prime.log"),
                        cfg.get("logging", "level",    default="INFO"))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = set(args.ext.split(","))
    source = Path(args.source)
    video_files = ([source] if source.is_file()
                   else [f for f in source.rglob("*") if f.suffix in extensions])

    if not video_files:
        logger.error(f"No video files found at {source}")
        sys.exit(1)

    logger.info(f"Found {len(video_files)} video(s) — output → {output_dir}")

    model_path = cfg.get("yolo", "model_path", default="models/yolo/finetuned/best.pt")
    yolo_model  = YOLO(model_path)
    farneback   = FarnebackFlow(cfg)
    crop_builder = CropBuilder(cfg)

    total_saved = 0
    for vf in video_files:
        logger.info(f"Processing {vf.name}")
        n = collect_from_video(vf, output_dir, yolo_model, farneback,
                               crop_builder, cfg, logger)
        logger.info(f"  → {n} crops saved")
        total_saved += n

    logger.info(f"Done — {total_saved} total crops → {output_dir}")
    logger.info(f"Next: python scripts/label_crops.py --input {output_dir}")


if __name__ == "__main__":
    main()
