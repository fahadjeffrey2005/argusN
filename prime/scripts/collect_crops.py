"""
PRIME — collect_crops.py
Run YOLO + Farneback flow on all videos and save every flagged
candidate as a 4-channel crop (BGR + flow magnitude) to raw_crops/.

Usage:
    python scripts/collect_crops.py \
      --source data/raw/videos/ \
      --output data/crops/raw_crops \
      --config config/config.yaml

Each saved crop is a 4-channel PNG: (crop_size, crop_size, 4).
Filename encodes video, frame index, and candidate index:
    <video_stem>_f<frame_idx>_c<candidate_idx>.png
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.config_loader import load_config
from src.detection.yolo_detector import YOLODetector
from src.flow.farneback import FarnebackFlow
from src.flow.egomotion import Egomotion
from src.flow.residual import FlowResidual
from src.fusion.prime_fusion import PrimeFusion
from src.semantic.crop_builder import CropBuilder


def collect_from_video(
    video_path: Path,
    output_dir: Path,
    yolo: YOLODetector,
    flow: FarnebackFlow,
    ego: Egomotion,
    residual: FlowResidual,
    fusion: PrimeFusion,
    crop_builder: CropBuilder,
    top_crop: float,
    bot_crop: float
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stem = video_path.stem
    saved = 0
    flow.reset()

    for frame_idx in tqdm(range(total_frames), desc=f"  {stem}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        # ROI crop
        h = frame.shape[0]
        y_start = int(h * top_crop)
        y_end = int(h * (1 - bot_crop))
        frame_roi = frame[y_start:y_end, :]

        # Farneback flow
        raw_flow = flow.compute(frame_roi)
        if raw_flow is None:
            continue

        flow_mag = flow.flow_magnitude(raw_flow)

        # Egomotion expected flow
        expected = ego.compute_expected_flow()
        # Resize expected to match ROI dimensions
        roi_h, roi_w = frame_roi.shape[:2]
        expected_roi = cv2.resize(expected, (roi_w, roi_h))

        # Residual candidates
        _, _, flow_candidates = residual.compute(raw_flow, expected_roi)

        # YOLO candidates
        yolo_candidates = yolo.detect(frame_roi)

        # Merge
        merged = fusion.merge(yolo_candidates, flow_candidates)

        # Build and save crops
        for ci, (crop, candidate) in enumerate(
            crop_builder.build_batch(frame_roi, flow_mag, merged)
        ):
            if crop is None:
                continue
            filename = f"{stem}_f{frame_idx:06d}_c{ci:03d}.png"
            save_path = output_dir / filename
            crop_builder.save_crop(crop, str(save_path))
            saved += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Collect 4-channel crops from videos")
    parser.add_argument("--source", required=True, help="Directory of video files or single video")
    parser.add_argument("--output", default="data/crops/raw_crops", help="Output directory for crops")
    parser.add_argument("--config", default="config/config.yaml", help="Config path")
    parser.add_argument("--ext", default=".mp4,.avi,.mov", help="Video extensions to process")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = set(args.ext.split(","))
    source = Path(args.source)

    if source.is_file():
        video_files = [source]
    else:
        video_files = [f for f in source.rglob("*") if f.suffix.lower() in extensions]

    if not video_files:
        print(f"No video files found in {source}")
        sys.exit(1)

    print(f"Found {len(video_files)} video(s). Output → {output_dir}")

    yolo = YOLODetector(cfg)
    flow = FarnebackFlow(cfg)
    ego = Egomotion(cfg)
    residual = FlowResidual(cfg)
    fusion = PrimeFusion(cfg)
    crop_builder = CropBuilder(cfg)

    top_crop = cfg.get("pipeline", "top_crop", default=0.22)
    bot_crop = cfg.get("pipeline", "bot_crop", default=0.15)

    total_saved = 0
    for vf in video_files:
        print(f"\nProcessing: {vf.name}")
        n = collect_from_video(
            vf, output_dir, yolo, flow, ego, residual, fusion, crop_builder,
            top_crop, bot_crop
        )
        print(f"  → {n} crops saved")
        total_saved += n

    print(f"\nDone. Total crops saved: {total_saved} → {output_dir}")


if __name__ == "__main__":
    main()
