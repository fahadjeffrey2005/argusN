#!/bin/bash
# ============================================================
# ARGUS-N — Data Organization & Cleanup Script
#
# Run ONCE from the argusN root:
#   cd /Volumes/T72/argusN
#   bash organize_argusN.sh
#
# What this does:
#   1. Extracts clean tarmac frames from clean1.mp4
#   2. Moves all raw videos into yolofinetune/data/raw/videos/
#   3. Copies 100 clean frames into hawkeye/data/clean_frames/ (PatchCore bank)
#   4. Moves first_finetuned_model.pt into yolofinetune/models/yolo/
#   5. Deletes everything not needed by any model
# ============================================================

set -e

ROOT="/Volumes/T72/argusN"
YOLO="$ROOT/yolofinetune"
HAWKEYE="$ROOT/hawkeye"
PRIME="$ROOT/prime"

echo ""
echo "========================================"
echo "  ARGUS-N Data Organization"
echo "========================================"
echo ""

# ── 1. Check Python and OpenCV ───────────────────────────────
echo "[1/6] Checking dependencies..."
PYTHON=$(which python3)
$PYTHON -c "import cv2" 2>/dev/null || {
    echo "  OpenCV not found in system Python."
    echo "  Activating yolofinetune venv..."
    source "$YOLO/venv/bin/activate" 2>/dev/null || {
        echo "  ERROR: venv not set up yet. Run 'bash $YOLO/setup.sh' first."
        exit 1
    }
    PYTHON="$YOLO/venv/bin/python3"
}
echo "  Python: $($PYTHON --version)"
echo "  OpenCV: $($PYTHON -c 'import cv2; print(cv2.__version__)')"

# ── 2. Create destination directories ───────────────────────
echo ""
echo "[2/6] Creating directories..."
mkdir -p "$YOLO/data/raw/videos/fod_sessions"
mkdir -p "$YOLO/data/raw/images"
mkdir -p "$HAWKEYE/data/clean_frames"
mkdir -p "$HAWKEYE/data/raw/videos"
mkdir -p "$PRIME/data/raw/videos"
echo "  Done."

# ── 3. Extract frames from clean1.mp4 ────────────────────────
echo ""
echo "[3/6] Extracting clean frames from clean1.mp4..."
CLEAN_VIDEO="$ROOT/raw data/clean1.mp4"
CLEAN_OUT="$YOLO/data/raw/images"

if [ ! -f "$CLEAN_VIDEO" ]; then
    echo "  WARNING: clean1.mp4 not found at '$CLEAN_VIDEO' — skipping"
else
    $PYTHON - <<PYEOF
import cv2
import os
import math
from pathlib import Path

video_path = "$CLEAN_VIDEO"
output_dir = Path("$CLEAN_OUT")
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"  ERROR: Could not open {video_path}")
    exit(1)

src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
target_fps = 2.0
interval = max(1, round(src_fps / target_fps))

print(f"  Source : {src_fps:.1f} fps, {total} frames ({total/src_fps:.1f}s)")
print(f"  Extract: every {interval} frames → ~{src_fps/interval:.1f} fps")

saved = 0
idx   = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % interval == 0:
        out_path = output_dir / f"clean1_{idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1
    idx += 1

cap.release()
print(f"  Saved  : {saved} frames → {output_dir}")
PYEOF

    echo "  clean1.mp4 frames extracted."
fi

# ── 4. Copy 100 evenly-spaced clean frames to hawkeye/data/clean_frames/ ──
echo ""
echo "[4/6] Copying 100 clean frames for PatchCore bank → hawkeye/data/clean_frames/..."
HAWKEYE_CLEAN="$HAWKEYE/data/clean_frames"

$PYTHON - <<PYEOF
import cv2
import os
from pathlib import Path

video_path = "$CLEAN_VIDEO"
output_dir = Path("$HAWKEYE_CLEAN")
output_dir.mkdir(parents=True, exist_ok=True)

if not Path(video_path).exists():
    print("  WARNING: clean1.mp4 not found — skipping hawkeye clean frames")
    exit(0)

cap = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n_frames = 100
step = max(1, total // n_frames)

saved = 0
for i in range(n_frames):
    target = i * step
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    if not ret:
        break
    out_path = output_dir / f"clean_bank_{i:03d}.jpg"
    cv2.imwrite(str(out_path), frame)
    saved += 1

cap.release()
print(f"  Saved  : {saved} frames → {output_dir}")
PYEOF

# Same 100 frames for prime
cp "$HAWKEYE/data/clean_frames/"*.jpg "$PRIME/data/raw/" 2>/dev/null || true
mkdir -p "$PRIME/data/clean_frames"
cp "$HAWKEYE/data/clean_frames/"*.jpg "$PRIME/data/clean_frames/" 2>/dev/null || true
echo "  Done."

# ── 5. Move videos into model directories ────────────────────
echo ""
echo "[5/6] Moving video files..."

# clean1.mp4 → yolofinetune + hawkeye + prime raw videos
cp "$ROOT/raw data/clean1.mp4" "$YOLO/data/raw/videos/clean_runway.mp4" && \
    echo "  clean1.mp4 → yolofinetune/data/raw/videos/clean_runway.mp4"
cp "$ROOT/raw data/clean1.mp4" "$HAWKEYE/data/raw/videos/clean_runway.mp4" && \
    echo "  clean1.mp4 → hawkeye/data/raw/videos/clean_runway.mp4"
cp "$ROOT/raw data/clean1.mp4" "$PRIME/data/raw/videos/clean_runway.mp4" && \
    echo "  clean1.mp4 → prime/data/raw/videos/clean_runway.mp4"

# fod1.mp4 → yolofinetune fod sessions
if [ -f "$ROOT/raw data/fod1.mp4" ]; then
    cp "$ROOT/raw data/fod1.mp4" "$YOLO/data/raw/videos/fod_sessions/fod1.mp4" && \
        echo "  fod1.mp4 → yolofinetune/data/raw/videos/fod_sessions/"
fi

# fod_16sep recordings → yolofinetune fod sessions
for f in "$ROOT/raw data/trained data/fod_16sep"*.mp4; do
    [ -f "$f" ] && cp "$f" "$YOLO/data/raw/videos/fod_sessions/" && \
        echo "  $(basename "$f") → yolofinetune/data/raw/videos/fod_sessions/"
done

# runway_corner_16sep.mp4
if [ -f "$ROOT/raw data/trained data/runway_corner_16sep.mp4" ]; then
    cp "$ROOT/raw data/trained data/runway_corner_16sep.mp4" "$YOLO/data/raw/videos/fod_sessions/" && \
        echo "  runway_corner_16sep.mp4 → yolofinetune/data/raw/videos/fod_sessions/"
fi

# recording_*.mp4 → yolofinetune (may contain usable footage)
for f in "$ROOT/raw data/recording_"*.mp4; do
    [ -f "$f" ] && cp "$f" "$YOLO/data/raw/videos/" && \
        echo "  $(basename "$f") → yolofinetune/data/raw/videos/"
done
for f in "$ROOT/raw data/arbitary/recording_"*.mp4; do
    [ -f "$f" ] && cp "$f" "$YOLO/data/raw/videos/" && \
        echo "  $(basename "$f") → yolofinetune/data/raw/videos/"
done

# first_finetuned_model.pt → yolofinetune/models/yolo/
if [ -f "$ROOT/raw data/first_finetuned_model.pt" ]; then
    cp "$ROOT/raw data/first_finetuned_model.pt" "$YOLO/models/yolo/first_finetuned_model.pt" && \
        echo "  first_finetuned_model.pt → yolofinetune/models/yolo/"
fi

# yolov8n.pt → yolofinetune/models/yolo/ (if setup.sh hasn't done this yet)
if [ -f "$ROOT/models/yolo/yolov8n.pt" ] && [ ! -f "$YOLO/models/yolo/yolov8n.pt" ]; then
    cp "$ROOT/models/yolo/yolov8n.pt" "$YOLO/models/yolo/yolov8n.pt" && \
        echo "  yolov8n.pt → yolofinetune/models/yolo/"
fi

echo "  Videos organized."

# ── 6. Delete files not needed by any model ─────────────────
echo ""
echo "[6/6] Deleting unused files..."

# data/replay_buffer/
rm -rf "$ROOT/data/replay_buffer" && echo "  DELETED data/replay_buffer/"

# data/synthetic/
rm -rf "$ROOT/data/synthetic" && echo "  DELETED data/synthetic/"

# models/raft/ — RAFT not used by any model (all use Farneback)
rm -rf "$ROOT/models/raft" && echo "  DELETED models/raft/"

# .ndjson annotation exports — raw data no longer needed
rm -f "$ROOT/data/runway-fod-2.ndjson" && echo "  DELETED data/runway-fod-2.ndjson"
rm -f "$ROOT/data/runway-fod-3.ndjson" && echo "  DELETED data/runway-fod-3.ndjson"

# download_ndjson_dataset.py — utility script no longer needed
rm -f "$ROOT/download_ndjson_dataset.py" && echo "  DELETED download_ndjson_dataset.py"

# corrupted .avi files
rm -rf "$ROOT/raw data/corrupted" && echo "  DELETED raw data/corrupted/"

# raw data/ staging directory — all useful content copied above
# NOTE: Only delete if you're confident copies are complete.
# Uncomment the lines below AFTER verifying the copies above succeeded:
# rm -rf "$ROOT/raw data"
# echo "  DELETED raw data/ (staging directory)"

# data/raw/images/ at root level — these were pre-extracted to wrong location
# They can be re-extracted if needed. Uncomment to remove:
# rm -rf "$ROOT/data/raw/images/"*.jpg
# echo "  DELETED old root data/raw/images/ frames"

echo ""
echo "========================================"
echo "  Organization complete."
echo ""
echo "  Summary:"
echo "  yolofinetune/data/raw/images/     ← clean background frames (no labels needed)"
echo "  yolofinetune/data/raw/videos/     ← all raw videos"
echo "  hawkeye/data/clean_frames/        ← 100 frames for PatchCore bank"
echo "  yolofinetune/models/yolo/         ← yolov8n.pt + first_finetuned_model.pt"
echo ""
echo "  Next steps:"
echo "  1. Run 'bash yolofinetune/setup.sh' if you haven't yet"
echo "  2. Extract FOD frames from fod sessions:"
echo "     python yolofinetune/scripts/extract_frames.py \\"
echo "       --video yolofinetune/data/raw/videos/fod_sessions/fod1.mp4 \\"
echo "       --output yolofinetune/data/annotated/images/train \\"
echo "       --fps 2"
echo "  3. Annotate with LabelImg:"
echo "     labelImg yolofinetune/data/annotated/images/train \\"
echo "              yolofinetune/data/annotated/labels/train"
echo "  4. Run augmentation, then train."
echo "========================================"
echo ""
