"""
prepare_dataset.py — Fix Roboflow export structure, remap classes, create train/val/test split.

Run once after downloading from Roboflow.

What it does:
  1. Finds images in Roboflow's staging layout (train/images/, valid/images/, etc.)
  2. Remaps all class IDs to 0 (fod) — fixes inconsistent Roboflow class names
  3. Splits into 70% train / 15% val / 15% test
  4. Writes into the correct ARGUS-N structure (images/{train,val,test}, labels/{train,val,test})
  5. Cleans up Roboflow's staging directories

Usage:
    python scripts/prepare_dataset.py
"""

import shutil
import random
from pathlib import Path


ANNOTATED_ROOT = Path("data/annotated")
SPLIT_RATIOS   = (0.70, 0.15, 0.15)   # train / val / test
RANDOM_SEED    = 42


def remap_label(src_path: Path, dst_path: Path):
    """
    Copy a YOLO label file, forcing all class IDs to 0 (fod).
    Roboflow had inconsistent names: 'Foreign Object', 'Foreign Objecy',
    'maybe', 'Hawkeye', '0' — all map to class 0 = fod.
    """
    lines = []
    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
    with open(dst_path, "w") as f:
        f.writelines(lines)


def collect_images(root: Path) -> list:
    """
    Collect all images from Roboflow's staging layout.
    Handles: train/images/, valid/images/, test/images/, or flat layout.
    """
    images = []
    for subdir in ["train", "valid", "test"]:
        staging = root / subdir / "images"
        if staging.exists():
            images += list(staging.glob("*.jpg"))
            images += list(staging.glob("*.png"))
            images += list(staging.glob("*.webp"))

    # Also check if images landed directly in images/train etc (already structured)
    for split in ["train", "val", "test"]:
        existing = root / "images" / split
        if existing.exists():
            for img in existing.glob("*.jpg"):
                if img not in images:
                    images.append(img)

    return images


def find_label(img_path: Path, root: Path) -> Path | None:
    """
    Find the label file for an image across Roboflow's staging directories.
    """
    stem = img_path.stem
    for subdir in ["train", "valid", "test"]:
        candidate = root / subdir / "labels" / f"{stem}.txt"
        if candidate.exists():
            return candidate
    return None


def main():
    root = ANNOTATED_ROOT
    if not root.exists():
        print(f"ERROR: {root} does not exist. Run from yolofinetune/ root.")
        return

    # ── Collect all images ───────────────────────────────────
    images = collect_images(root)
    print(f"Found {len(images)} images across all staging directories")

    if not images:
        print("No images found. Check that the Roboflow download completed.")
        return

    # ── Split ────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    random.shuffle(images)

    n       = len(images)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val   = int(n * SPLIT_RATIOS[1])

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    print(f"\nSplit ({SPLIT_RATIOS[0]:.0%}/{SPLIT_RATIOS[1]:.0%}/{SPLIT_RATIOS[2]:.0%}):")
    for split, imgs in splits.items():
        print(f"  {split:6s}: {len(imgs)} images")

    # ── Copy into ARGUS-N structure ──────────────────────────
    print("\nCopying and remapping labels...")

    no_label_count = 0
    for split, split_imgs in splits.items():
        img_dst = root / "images" / split
        lbl_dst = root / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            # Skip if image is already in the destination
            dst_img = img_dst / img_path.name
            if dst_img != img_path:
                shutil.copy(img_path, dst_img)

            # Find and remap label
            lbl_src = find_label(img_path, root)
            if lbl_src:
                remap_label(lbl_src, lbl_dst / (img_path.stem + ".txt"))
            else:
                # No label = background frame — YOLO treats unlabelled images as negatives
                no_label_count += 1

        # Write classes.txt
        (lbl_dst / "classes.txt").write_text("fod\n")

    if no_label_count:
        print(f"  Note: {no_label_count} images had no label file (treated as background)")

    # ── Verify ───────────────────────────────────────────────
    print("\nFinal counts:")
    total_imgs = 0
    total_lbls = 0
    for split in ["train", "val", "test"]:
        n_imgs = len(list((root / "images" / split).glob("*.jpg"))) + \
                 len(list((root / "images" / split).glob("*.png"))) + \
                 len(list((root / "images" / split).glob("*.webp")))
        n_lbls = len([f for f in (root / "labels" / split).glob("*.txt")
                      if f.name != "classes.txt"])
        print(f"  {split:6s}: {n_imgs} images, {n_lbls} label files")
        total_imgs += n_imgs
        total_lbls += n_lbls
    print(f"  {'TOTAL':6s}: {total_imgs} images, {total_lbls} label files")

    # ── Clean up Roboflow staging dirs ───────────────────────
    print("\nCleaning up Roboflow staging directories...")
    for item in ["train", "valid", "test", "data.yaml", "README.roboflow.txt"]:
        p = root / item
        if p.is_dir():
            shutil.rmtree(p)
            print(f"  Removed {item}/")
        elif p.is_file():
            p.unlink()
            print(f"  Removed {item}")

    print("\nDataset ready.")
    print(f"  Images : {root}/images/{{train,val,test}}/")
    print(f"  Labels : {root}/labels/{{train,val,test}}/")
    print("\nNext: run augmentation on the train split, then train.")
    print("  python scripts/augment_dataset.py --input data/annotated --output data/augmented --factor 5")


if __name__ == "__main__":
    main()
