#!/bin/bash
# ============================================================
# PRIME Setup — M4 Mac
# Creates Python venv entirely on the drive.
# Run once from the prime directory:
#   cd /Volumes/T72/argusN/prime
#   bash setup.sh
# ============================================================

set -e

DRIVE_PATH="/Volumes/T72/argusN/prime"
VENV_PATH="$DRIVE_PATH/venv"

echo ""
echo "========================================"
echo "  PRIME Environment Setup"
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
echo "[4/4] Installing packages (this takes a few minutes)..."
pip install -r "$DRIVE_PATH/requirements.txt"

echo ""
echo "========================================"
echo "  Setup complete."
echo ""
echo "  To activate:"
echo "  source /Volumes/T72/argusN/prime/venv/bin/activate"
echo ""
echo "  Next steps:"
echo "  1. Copy YOLO weights from yolofinetune:"
echo "     cp ../yolofinetune/models/yolo/finetuned/best.pt models/yolo/finetuned/best.pt"
echo "  2. Collect CNN training crops:"
echo "     python scripts/collect_crops.py --source data/raw/videos/ --output data/crops/raw_crops"
echo "  3. Label crops:"
echo "     python scripts/label_crops.py --input data/crops/raw_crops --output data/crops"
echo "  4. Train CNN:"
echo "     python scripts/train_cnn.py"
echo "  5. Run full pipeline:"
echo "     python scripts/run_prime.py"
echo "========================================"
echo ""
