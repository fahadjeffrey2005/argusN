"""
split_dataset.py — Split annotated dataset into train/val/test (70/15/15).

Run this after prepare_dataset.py if Roboflow downloaded everything into
train/ only with no val/test split applied.

Usage (run from inside prime/ directory):
    python scripts/split_dataset.py
"""

import shutil
import random
from pathlib import Path


def split_dataset(
    annotated_dir: str = "data/annotated",
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42
):
    annotated = Path(annotated_dir)
    img_train  = annotated / "images" / "train"
    lbl_train  = annotated / "labels" / "train"

    images = sorted(img_train.glob("*.jpg")) + sorted(img_train.glob("*.png"))
    if not images:
        print(f"No images found in {img_train}")
        return

    random.seed(seed)
    random.shuffle(images)

    n       = len(images)
    n_val   = int(n * val_ratio)
    n_test  = int(n * val_ratio)
    # n_train = remainder stays in train

    splits = {
        "val":  images[:n_val],
        "test": images[n_val:n_val + n_test],
    }

    for split, imgs in splits.items():
        (annotated / "images" / split).mkdir(parents=True, exist_ok=True)
        (annotated / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.move(str(img), annotated / "images" / split / img.name)
            lbl = lbl_train / (img.stem + ".txt")
            if lbl.exists():
                shutil.move(str(lbl), annotated / "labels" / split / lbl.name)

    print("Split complete:")
    for split in ["train", "val", "test"]:
        n_imgs = len(list((annotated / "images" / split).glob("*.jpg")) +
                     list((annotated / "images" / split).glob("*.png")))
        n_lbls = len(list((annotated / "labels" / split).glob("*.txt")))
        print(f"  {split:5s}: {n_imgs} images, {n_lbls} labels")


if __name__ == "__main__":
    split_dataset()
