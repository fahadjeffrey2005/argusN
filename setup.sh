#!/bin/bash
# ============================================================
# ARGUS-N Setup — M4 Mac
# Creates Python venv entirely on the drive.
# Run once from the project root:
#   cd /Volumes/T72/argusN
#   bash setup.sh
# ============================================================

set -e

DRIVE_PATH="/Volumes/T72/argusN"
VENV_PATH="$DRIVE_PATH/venv"

echo ""
echo "========================================"
echo "  ARGUS-N Environment Setup"
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
echo "  source /Volumes/T72/argusN/venv/bin/activate"
echo ""
echo "  To run pipeline:"
echo "  python scripts/run_pipeline.py --source path/to/video --visualise --speed 30"
echo "========================================"
echo ""
