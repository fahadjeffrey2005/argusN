"""
build_real_dataset.py
----------------------
Builds a clean single-class YOLO dataset from real data only. No synthetics.

Sources:
  1. Roboflow dataset (data/roboflow/) — 125 annotated images, 6 classes
     Remap: bolt(0), foreign object(1), oil marks(2), tire marks(5) → class 0 "fod"
     Skip:  runway_line(4), runway_damage(3) — not FOD
     Drop:  images where ALL annotations are runway_line/runway_damage (no real FOD)

  2. Received hard negatives (data/recieved/output/) — real patches from fod1.mp4
     clean_tarmac, runway_marking, shadow, strobe_light → empty labels (no FOD)
     These explicitly teach the model what NOT to fire on.

  NOTE: The 210 received FOD patches need manual bounding box annotation before
  they can be added. Upload them to Roboflow, annotate, then add to this dataset.
  That is the highest-value annotation task for the annotation team member.

Run from argusN/ on Ubuntu:
    python build_real_dataset.py \\
        --roboflow data/roboflow \\
        --received data/recieved \\
        --out data/real_single_class

Output:
    data/real_single_class/
        train/images/  train/labels/
        valid/images/  valid/labels/
        data.yaml      (nc=1, names=['fod'])
"""

import argparse
import random
import shutil
from pathlib import Path
import cv2
import numpy as np


# Roboflow class mapping:
# 0=bolt, 1=foreign object, 2=oil marks, 3=runway damage, 4=runway line, 5=tire marks
FOD_CLASSES    = {0, 1, 2, 5}   # remap these to class 0 "fod"
SKIP_CLASSES   = {3, 4}         # runway_damage, runway_line — skip these annotations


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--roboflow", default="data/roboflow")
    p.add_argument("--received", default="data/recieved")
    p.add_argument("--out",      default="data/real_single_class")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def remap_label_file(src_lbl: Path):
    """
    Read a Roboflow label file, remap FOD classes to 0, drop non-FOD classes.
    Returns list of remapped label lines, or None if no FOD annotations remain.
    """
    if not src_lbl.exists():
        return []
    lines = src_lbl.read_text().strip().split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if cls in FOD_CLASSES:
            out.append(f"0 {' '.join(parts[1:])}")
        # SKIP_CLASSES are silently dropped
    return out


def process_roboflow(roboflow_root: Path, out_root: Path, seed: int):
    """Process Roboflow dataset: remap classes, drop runway-only images."""
    print(f"\n[1] Processing Roboflow data: {roboflow_root}")
    counts = {"kept": 0, "dropped_no_fod": 0, "total_anns": 0}

    for split in ["train", "valid"]:
        src_img = roboflow_root / split / "images"
        src_lbl = roboflow_root / split / "labels"
        dst_img = out_root / split / "images"
        dst_lbl = out_root / split / "labels"

        if not src_img.exists():
            continue

        imgs = list(src_img.glob("*.jpg")) + list(src_img.glob("*.png"))
        for img_path in imgs:
            lbl_path = src_lbl / (img_path.stem + ".txt")
            remapped = remap_label_file(lbl_path)

            if not remapped:
                # Image has no real FOD annotations — skip
                counts["dropped_no_fod"] += 1
                continue

            # Copy image and write remapped labels
            shutil.copy2(img_path, dst_img / img_path.name)
            (dst_lbl / (img_path.stem + ".txt")).write_text("\n".join(remapped))
            counts["kept"] += 1
            counts["total_anns"] += len(remapped)

    print(f"    Kept   : {counts['kept']} images with real FOD annotations")
    print(f"    Dropped: {counts['dropped_no_fod']} images (runway line/damage only, no FOD)")
    print(f"    Total annotations: {counts['total_anns']}")
    return counts


def load_patches(folder: Path):
    """Load PNG patches, return list of BGR arrays."""
    patches = []
    if not folder.exists():
        return patches
    for f in sorted(folder.glob("*.png")):
        if f.name.startswith("._"):
            continue
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        patches.append((f.name, img))
    return patches


def add_hard_negatives(received_root: Path, out_root: Path, seed: int):
    """
    Add received negative patches as hard negatives with empty label files.
    Uses clean_tarmac, runway_marking, shadow, strobe_light patches.
    These are real images where the model must output ZERO detections.
    """
    print(f"\n[2] Adding hard negatives from received patches: {received_root}")
    neg_dirs = ["clean_tarmac", "runway_marking", "shadow", "strobe_light"]
    total = 0
    random.seed(seed)

    for folder_name in neg_dirs:
        patches = load_patches(received_root / "output" / folder_name)
        if not patches:
            print(f"    {folder_name}: not found, skipping")
            continue

        # Split 85/15
        random.shuffle(patches)
        split_idx = max(1, int(len(patches) * 0.85))
        splits = [("train", patches[:split_idx]), ("valid", patches[split_idx:])]

        for split, items in splits:
            dst_img = out_root / split / "images"
            dst_lbl = out_root / split / "labels"
            for fname, img in items:
                stem = f"neg_{folder_name}_{Path(fname).stem}"
                cv2.imwrite(str(dst_img / f"{stem}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                (dst_lbl / f"{stem}.txt").write_text("")  # empty = no FOD
                total += 1

        print(f"    {folder_name}: {len(patches)} hard negatives added")

    print(f"    Total hard negatives: {total}")
    return total


def main():
    args = parse_args()
    random.seed(args.seed)

    roboflow = Path(args.roboflow)
    received = Path(args.received)
    out      = Path(args.out)

    print(f"\n{'='*60}")
    print(f"  Hawkeye — Real Data Single-Class Dataset Builder")
    print(f"{'='*60}")
    print(f"  Roboflow : {roboflow}")
    print(f"  Received : {received}")
    print(f"  Output   : {out}")
    print(f"  Classes  : 1 (class 0 = 'fod')")
    print(f"  No synthetic data — real annotations only")
    print(f"{'='*60}")

    # Create output dirs
    for split in ["train", "valid"]:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    # 1. Remap Roboflow
    if roboflow.exists():
        process_roboflow(roboflow, out, args.seed)
    else:
        print(f"\n  WARNING: Roboflow data not found at {roboflow}")
        print(f"  Run: python download_roboflow.py --key YOUR_KEY")

    # 2. Add hard negatives from received patches
    if received.exists():
        add_hard_negatives(received, out, args.seed)
    else:
        print(f"\n  WARNING: Received data not found at {received}")

    # Count final dataset
    train_imgs = list((out / "train" / "images").glob("*.jpg"))
    valid_imgs = list((out / "valid" / "images").glob("*.jpg"))

    # 3. Write data.yaml
    yaml = f"""# Hawkeye Real Data — Single Class
# No synthetic data. Real annotations only.
# Source: Roboflow (remapped to single fod class) + received hard negatives
#
# TO ADD MORE DATA (highest priority):
#   1. Annotate the 210 FOD patches from data/recieved/output/fod/
#      Upload to Roboflow, draw bounding boxes, export and add here.
#   2. This is the single most impactful thing the annotation team can do.

path: {out.resolve()}
train: train/images
val:   valid/images

nc: 1
names: ['fod']
"""
    (out / "data.yaml").write_text(yaml)

    print(f"\n{'='*60}")
    print(f"  Dataset built!")
    print(f"  Train images: {len(train_imgs)}")
    print(f"  Valid images: {len(valid_imgs)}")
    print(f"  Total       : {len(train_imgs) + len(valid_imgs)}")
    print(f"  YAML        : {out}/data.yaml")
    print(f"\n  RETRAIN:")
    print(f"    python retrain.py --real {out} --upgrade --device cuda --name retrain_real_v1")
    print(f"\n  ANNOTATION PRIORITY:")
    print(f"    Upload data/recieved/output/fod/ (210 images) to Roboflow")
    print(f"    Annotate bounding boxes → adds 210 real camera-specific FOD images")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
