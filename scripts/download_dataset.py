"""
ARGUS-N Dataset Downloader
Downloads public FOD dataset from Roboflow Universe.
Saves to /data/raw/ on the drive.

Usage:
    python scripts/download_dataset.py

Requires:
    pip install roboflow  (already in requirements)
"""

import sys
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
DRIVE_ROOT   = Path(__file__).resolve().parent.parent
DATA_RAW     = DRIVE_ROOT / "data" / "raw"
DATA_IMAGES  = DATA_RAW / "images"
DATA_LABELS  = DATA_RAW / "labels"
DATA_YAML    = DATA_RAW / "data.yaml"

DATA_RAW.mkdir(parents=True, exist_ok=True)


def download_via_roboflow():
    """
    Download FOD dataset using Roboflow SDK.
    Dataset: foreignobjectaerodromes/fod-i2kfx
    Format: YOLOv8
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow not installed.")
        print("        Run: pip install roboflow")
        sys.exit(1)

    print("\n========================================")
    print("  ARGUS-N Dataset Download")
    print("========================================\n")

    # Public dataset — no API key needed for download
    # If it prompts for a key, press Enter to use anonymous access
    rf = Roboflow(api_key="")

    print("[1/3] Connecting to Roboflow Universe...")
    project = rf.workspace("foreignobjectaerodromes").project("fod-i2kfx")

    print("[2/3] Downloading dataset (YOLOv8 format)...")
    dataset = project.version(1).download(
        "yolov8",
        location=str(DATA_RAW),
        overwrite=True
    )

    print(f"[3/3] Dataset saved to: {DATA_RAW}")
    print(f"      Location: {dataset.location}")
    print("\nDone. Run the pipeline with:")
    print(f"  python scripts/run_pipeline.py --source {DATA_RAW}/images --visualise\n")

    return dataset.location


def fallback_instructions():
    """
    Print manual download instructions if Roboflow SDK fails.
    """
    print("\n[FALLBACK] Manual download instructions:")
    print("─" * 50)
    print("1. Open in browser:")
    print("   https://universe.roboflow.com/foreignobjectaerodromes/fod-i2kfx")
    print("")
    print("2. Click 'Download Dataset'")
    print("3. Select format: YOLOv8")
    print("4. Download and extract to:")
    print(f"   {DATA_RAW}/")
    print("")
    print("5. Then run:")
    print("   python scripts/run_pipeline.py --source data/raw/images --visualise")
    print("─" * 50)


if __name__ == "__main__":
    try:
        download_via_roboflow()
    except Exception as e:
        print(f"\n[WARNING] Roboflow download failed: {e}")
        fallback_instructions()
