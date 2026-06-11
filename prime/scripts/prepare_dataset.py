"""
PRIME — prepare_dataset.py
Downloads the annotated FOD dataset from Roboflow and prepares it for evaluation.
Fixes directory structure, remaps all classes to 0=fod, confirms 70/15/15 split.

Run from inside prime/:
    python scripts/prepare_dataset.py --dest data/annotated

Roboflow details:
    Workspace : durvas-workspace-ihhkq
    Project   : hawkeye-ap3a8
    Version   : 1
    Format    : yolov8
"""

import argparse
import shutil
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def remap_labels(label_dir: Path):
    """Force all class IDs in every label file to 0 (fod)."""
    label_files = list(label_dir.rglob("*.txt"))
    for lf in label_files:
        lines = lf.read_text().strip().splitlines()
        fixed = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                parts[0] = "0"
                fixed.append(" ".join(parts))
        lf.write_text("\n".join(fixed) + "\n" if fixed else "")
    print(f"  Remapped {len(label_files)} label files → class 0")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare PRIME dataset")
    parser.add_argument("--dest", default="data/annotated",
                        help="Destination directory for prepared dataset")
    parser.add_argument("--api-key", default="WCIxuet94KXWxzmAgRSQ",
                        help="Roboflow API key")
    parser.add_argument("--workspace", default="durvas-workspace-ihhkq")
    parser.add_argument("--project", default="hawkeye-ap3a8")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, just fix structure of existing data")
    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    tmp_dir = dest.parent / "_roboflow_tmp"

    # ── Download from Roboflow ─────────────────────────────
    if not args.skip_download:
        try:
            from roboflow import Roboflow
        except ImportError:
            print("ERROR: roboflow not installed. Run: pip install roboflow")
            sys.exit(1)

        print(f"Downloading from Roboflow...")
        rf = Roboflow(api_key=args.api_key)
        project = rf.workspace(args.workspace).project(args.project)
        version = project.version(args.version)
        version.download("yolov8", location=str(tmp_dir))
        print(f"Downloaded to {tmp_dir}")
    else:
        tmp_dir = dest.parent / "_roboflow_tmp"
        if not tmp_dir.exists():
            print(f"ERROR: --skip-download set but {tmp_dir} does not exist")
            sys.exit(1)

    # ── Find the downloaded project folder ────────────────
    # Roboflow creates a subfolder named after the project
    subdirs = [d for d in tmp_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        rf_root = subdirs[0]
    else:
        rf_root = tmp_dir  # fallback

    print(f"Roboflow root: {rf_root}")

    # ── Build canonical structure ──────────────────────────
    # prime expects:
    #   data/annotated/images/{train,val,test}/
    #   data/annotated/labels/{train,val,test}/

    split_map = {
        "train": "train",
        "valid": "val",
        "test":  "test",
    }

    for rf_split, canonical_split in split_map.items():
        for kind in ("images", "labels"):
            src = rf_root / rf_split / kind
            dst = dest / kind / canonical_split
            dst.mkdir(parents=True, exist_ok=True)

            if not src.exists():
                print(f"  WARNING: {src} not found — skipping")
                continue

            files = list(src.iterdir())
            for f in files:
                shutil.copy2(f, dst / f.name)
            print(f"  Copied {len(files)} files: {rf_split}/{kind} → {kind}/{canonical_split}/")

    # ── Remap all classes to 0 ─────────────────────────────
    print("Remapping classes...")
    remap_labels(dest / "labels")

    # ── Counts ────────────────────────────────────────────
    print("\nDataset summary:")
    for split in ("train", "val", "test"):
        img_count = len(list((dest / "images" / split).glob("*")))
        lbl_count = len(list((dest / "labels" / split).glob("*.txt")))
        print(f"  {split:5s}: {img_count} images, {lbl_count} labels")

    # ── Write data.yaml ───────────────────────────────────
    yaml_path = dest / "data.yaml"
    yaml_path.write_text(
        f"path: {dest}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"nc: 1\n"
        f"names: ['fod']\n"
    )
    print(f"\ndata.yaml written → {yaml_path}")

    # ── Cleanup tmp ───────────────────────────────────────
    if not args.skip_download and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        print(f"Cleaned up {tmp_dir}")

    print("\nDataset ready.")


if __name__ == "__main__":
    main()
