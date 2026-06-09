#!/bin/bash
# ============================================================
# ARGUS-N Setup — M4 Mac
# [YOLOFINETUNE — paths set to yolofinetune directory]
# Creates Python venv entirely on the drive.
# Run once from the project root:
#   cd /Volumes/T72/argusN/yolofinetune
#   bash setup.sh
# ============================================================

set -e

DRIVE_PATH="/Volumes/T72/argusN/yolofinetune"
VENV_PATH="$DRIVE_PATH/venv"

echo ""
echo "========================================"
echo "  YOLOFINETUNE Environment Setup"
echo "========================================"
echo ""

# ── Check Python ────────────────────────────────────────────
echo "[1/5] Checking Python..."
PYTHON=$(which python3)
PYTHON_VERSION=$($PYTHON --version 2>&1)
echo "      Using: $PYTHON ($PYTHON_VERSION)"

# ── Create venv on drive ─────────────────────────────────────
echo ""
echo "[2/5] Creating virtual environment on drive..."
if [ -d "$VENV_PATH" ]; then
    echo "      venv already exists at $VENV_PATH — skipping creation"
else
    $PYTHON -m venv "$VENV_PATH"
    echo "      venv created at $VENV_PATH"
fi

# ── Activate and upgrade pip ─────────────────────────────────
echo ""
echo "[3/5] Upgrading pip..."
source "$VENV_PATH/bin/activate"
pip install --upgrade pip --quiet

# ── Install all packages to drive ───────────────────────────
echo ""
echo "[4/5] Installing packages to drive (this takes a few minutes)..."
pip install -r "$DRIVE_PATH/requirements.txt"

# ── Copy pretrained YOLO weights if not present ─────────────
echo ""
echo "[5/5] Checking for yolov8n.pt..."
YOLO_SRC="/Volumes/T72/argusN/models/yolo/yolov8n.pt"
YOLO_DST="$DRIVE_PATH/models/yolo/yolov8n.pt"
if [ -f "$YOLO_DST" ]; then
    echo "      yolov8n.pt already present — skipping"
elif [ -f "$YOLO_SRC" ]; then
    cp "$YOLO_SRC" "$YOLO_DST"
    echo "      Copied yolov8n.pt from $YOLO_SRC"
else
    echo "      WARNING: yolov8n.pt not found at $YOLO_SRC"
    echo "      Download manually: https://github.com/ultralytics/assets/releases"
fi

echo ""
echo "========================================"
echo "  Setup complete."
echo ""
echo "  To activate:"
echo "  source /Volumes/T72/argusN/yolofinetune/venv/bin/activate"
echo ""
echo "  Workflow:"
echo "  Step 1 — Extract frames from clean video:"
echo "  python scripts/extract_frames.py --video data/raw/videos/recording.mp4 --output data/raw/images --fps 2"
echo ""
echo "  Step 2 — Annotate with LabelImg, then run augmentation:"
echo "  python scripts/augment_dataset.py --input data/annotated --output data/augmented --factor 5"
echo ""
echo "  Step 3 — Train:"
echo "  python scripts/train_yolo.py"
echo ""
echo "  Step 4 — Evaluate:"
echo "  python scripts/evaluate.py --model models/yolo/finetuned/best.pt --data data/annotated/test"
echo "========================================"
echo ""
