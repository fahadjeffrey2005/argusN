#!/bin/bash
# ARGUS-N — Ubuntu one-shot: patch + run pipeline
set -e
cd ~/argusN
source venv/bin/activate
echo "==> Patching Pathway D..."
python scripts/patch_pathway_d.py
echo "==> Downloading dataset..."
python scripts/download_ndjson_dataset.py --files runway-fod-2.ndjson runway-fod-3.ndjson --out data/yolo_dataset --workers 16
echo "==> Running pipeline..."
python scripts/run_pipeline.py --source "raw data/recording_20250521_141904.mp4" --build-bank --floor-crop 0.35 --pc-threshold 0.3 --patchcore-stride 3 --clip-stride 5
