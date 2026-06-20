"""
merge_staging.py — Merge reviewed staging data into the main YOLO dataset.
===========================================================================

Run this AFTER you've reviewed and corrected all labels in the staging folder.
It adds new images to train (80%), val (10%), test (10%) splits.

Usage
-----
    python merge_staging.py \
        --staging data/staging_new \
        --dataset data/yolo_dataset

    # Dry run (shows what would happen, doesn't copy anything):
    python merge_staging.py --staging data/staging_new --dataset data/yolo_dataset --dry-run
"""

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge staged annotations into main dataset")
    parser.add_argument("--staging", required=True, help="Path to staging_new folder")
    parser.add_argument("--dataset", required=True, help="Path to yolo_dataset folder")
    parser.add_argument("--split",   nargs=3, type=float, default=[0.80, 0.10, 0.10],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Train/val/test split ratios (default: 0.80 0.10 0.10)")
    parser.add_argument("--seed",    type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without copying")
    args = parser.parse_args()

    staging = Path(args.staging)
    dataset = Path(args.dataset)

    images_src = staging / "images"
    labels_src = staging / "labels"

    if not images_src.exists():
        raise FileNotFoundError(f"No images/ folder in {staging}. Run prepare_new_data.py first.")

    # Collect all annotated images (those with a corresponding label file)
    all_images = sorted(images_src.glob("*.jpg"))
    paired = [(img, labels_src / (img.stem + ".txt")) for img in all_images]
    # Only include images that have a label file (even empty ones count — clean frames)
    paired = [(img, lbl) for img, lbl in paired if lbl.exists()]

    total = len(paired)
    if total == 0:
        print("No paired image+label files found. Nothing to merge.")
        return

    random.seed(args.seed)
    random.shuffle(paired)

    n_train = int(total * args.split[0])
    n_val   = int(total * args.split[1])
    n_test  = total - n_train - n_val

    splits = {
        "train": paired[:n_train],
        "val":   paired[n_train:n_train + n_val],
        "test":  paired[n_train + n_val:],
    }

    print(f"\nMerge plan: {total} files → train={n_train}, val={n_val}, test={n_test}")
    if args.dry_run:
        print("DRY RUN — no files will be copied\n")

    copied = 0
    for split_name, pairs in splits.items():
        img_dst = dataset / "images" / split_name
        lbl_dst = dataset / "labels" / split_name
        if not args.dry_run:
            img_dst.mkdir(parents=True, exist_ok=True)
            lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in pairs:
            dst_img = img_dst / img_path.name
            dst_lbl = lbl_dst / lbl_path.name
            if args.dry_run:
                print(f"  [{split_name}] {img_path.name}")
            else:
                shutil.copy2(img_path, dst_img)
                shutil.copy2(lbl_path, dst_lbl)
                copied += 1

    if not args.dry_run:
        print(f"\nDone — {copied} image+label pairs merged into {dataset}")
        print("\nNext: fine-tune from existing best.pt checkpoint:")
        print("  python yolofinetune/scripts/train.py  # or resume training")
    else:
        print(f"\nDry run complete. Run without --dry-run to copy {total} files.")


if __name__ == "__main__":
    main()
