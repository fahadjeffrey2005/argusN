#!/bin/bash
# ============================================================
# ARGUS-N Setup — M4 Mac
# [HAWKEYE copy — paths updated to hawkeye directory]
# Creates Python venv entirely on the drive.
# Run once from the project root:
#   cd /Volumes/T72/argusN/hawkeye
#   bash setup.sh
# ============================================================

set -e

DRIVE_PATH="/Volumes/T72/argusN/hawkeye"
VENV_PATH="$DRIVE_PATH/venv"

echo ""
echo "========================================"
echo "  HAWKEYE Environment Setup"
echo "========================================"
echo ""

# ── Check Python ────────────────────────────────────────────
echo "[1/4] Checking Python..."
PYTHON=$(which python3)
PYTHON_VERSION=$($PYTHON --version 2>&1)
echo "      Using: $PYTHON ($PYTHON_VERSION)"

# ── Create venv on drive ─────────────────────────────────────
echo ""
echo "[2/4] Creating virtual environment on drive..."
if [ -d "$VENV_PATH" ]; then
    echo "      venv already exists at $VENV_PATH — skipping creation"
else
    $PYTHON -m venv "$VENV_PATH"
    echo "      venv created at $VENV_PATH"
fi

# ── Activate and upgrade pip ─────────────────────────────────
echo ""
echo "[3/4] Upgrading pip..."
source "$VENV_PATH/bin/activate"
pip install --upgrade pip --quiet

# ── Install all packages to drive ───────────────────────────
echo ""
echo "[4/4] Installing packages to drive (this takes a few minutes)..."
pip install -r "$DRIVE_PATH/requirements.txt"

echo ""
echo "========================================"
echo "  Setup complete."
echo ""
echo "  To activate:"
echo "  source /Volumes/T72/argusN/hawkeye/venv/bin/activate"
echo ""
echo "  Step 1 — Copy YOLO weights from yolofinetune (after training):"
echo "  cp ../yolofinetune/models/yolo/finetuned/best.pt models/yolo/finetuned/best.pt"
echo ""
echo "  Step 2 — Build PatchCore bank:"
echo "  python scripts/build_patchcore_bank.py --video data/raw/videos/clean_runway.mp4 --frames 100"
echo ""
echo "  Step 3 — Run pipeline:"
echo "  python scripts/run_hawkeye.py --visualise --speed 30"
echo "========================================"
echo ""
