"""
augment_dataset.py — Augment annotated FOD dataset to reach training target.

For each annotated image+label pair, generates `factor` augmented copies using:
  - Horizontal flip
  - Brightness/contrast shift ±30%
  - Gaussian noise
  - Copy-paste: paste FOD crop onto a different clean tarmac background

All augmented copies maintain valid YOLO label format: class_id cx cy w h (normalised).

Usage:
    python scripts/augment_dataset.py \
        --input data/annotated \
        --output data/augmented \
        --factor 5

    # Separate augmentation per split:
    python scripts/augment_dataset.py \
        --input data/annotated \
        --output data/augmented \
        --split train \
        --factor 5
"""

import cv2
import numpy as np
import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm


# ── Augmentation helpers ────────────────────────────────────────────────────

def flip_horizontal(image: np.ndarray, labels: list) -> tuple:
    """Flip image horizontally and adjust YOLO cx coordinates."""
    flipped = cv2.flip(image, 1)
    new_labels = []
    for label in labels:
        cls, cx, cy, w, h = label
        new_labels.append([cls, 1.0 - cx, cy, w, h])
    return flipped, new_labels


def adjust_brightness(image: np.ndarray, labels: list, factor_range: tuple = (0.7, 1.3)) -> tuple:
    """Multiply pixel values by a random brightness factor."""
    factor = random.uniform(*factor_range)
    adjusted = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return adjusted, labels  # labels unchanged


def add_gaussian_noise(image: np.ndarray, labels: list, sigma: float = 15.0) -> tuple:
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy, labels  # labels unchanged


def copy_paste_fod(
    image: np.ndarray,
    labels: list,
    clean_backgrounds: list,
    min_scale: float = 0.8,
    max_scale: float = 1.2
) -> tuple:
    """
    Paste FOD crop(s) from the current image onto a randomly selected clean background.
    Returns the composited image with updated labels.
    """
    if not clean_backgrounds:
        return image, labels

    h, w = image.shape[:2]
    bg_path = random.choice(clean_backgrounds)
    bg = cv2.imread(str(bg_path))
    if bg is None:
        return image, labels
    bg = cv2.resize(bg, (w, h))

    new_labels = []
    for label in labels:
        cls, cx, cy, bw, bh = label

        # Extract FOD crop with padding
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Scale crop randomly
        scale = random.uniform(min_scale, max_scale)
        new_cw = max(1, int((x2 - x1) * scale))
        new_ch = max(1, int((y2 - y1) * scale))
        crop_resized = cv2.resize(crop, (new_cw, new_ch))

        # Place at a random valid position
        max_px = w - new_cw
        max_py = h - new_ch
        if max_px <= 0 or max_py <= 0:
            continue
        px = random.randint(0, max_px)
        py = random.randint(0, max_py)

        bg[py:py + new_ch, px:px + new_cw] = crop_resized

        # Recompute normalised label
        new_cx = (px + new_cw / 2) / w
        new_cy = (py + new_ch / 2) / h
        new_bw = new_cw / w
        new_bh = new_ch / h
        new_labels.append([cls, new_cx, new_cy, new_bw, new_bh])

    if not new_labels:
        return image, labels

    return bg, new_labels


# ── Label I/O ───────────────────────────────────────────────────────────────

def load_labels(label_path: Path) -> list:
    """Load YOLO format labels. Returns list of [cls, cx, cy, w, h]."""
    if not label_path.exists():
        return []
    labels = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return labels


def save_labels(labels: list, label_path: Path):
    """Save labels in YOLO format."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for label in labels:
            cls, cx, cy, w, h = label
            f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ── Augmentation pipeline ────────────────────────────────────────────────────

AUGMENTATIONS = [
    ("flip", flip_horizontal),
    ("bright", adjust_brightness),
    ("noise", add_gaussian_noise),
]


def augment_split(
    input_images: Path,
    input_labels: Path,
    output_images: Path,
    output_labels: Path,
    factor: int,
    clean_backgrounds: list,
    copy_first: bool = True
):
    """
    Augment all images in a split directory.
    copy_first: if True, copy originals into output before augmenting.
    """
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(input_images.glob("*.jpg")) + list(input_images.glob("*.png")))

    if not image_paths:
        print(f"  No images found in {input_images}")
        return

    print(f"  Found {len(image_paths)} images. Augmenting x{factor}...")

    for img_path in tqdm(image_paths, desc=f"  {input_images.name}"):
        label_path = input_labels / (img_path.stem + ".txt")
        labels = load_labels(label_path)

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Copy original
        if copy_first:
            shutil.copy(img_path, output_images / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, output_labels / label_path.name)

        # Generate augmented copies
        aug_funcs = AUGMENTATIONS.copy()
        # Add copy-paste if clean backgrounds available
        if clean_backgrounds:
            aug_funcs.append(("copypaste", lambda i, l: copy_paste_fod(i, l, clean_backgrounds)))

        for i in range(factor):
            aug_name, aug_fn = random.choice(aug_funcs)
            aug_img, aug_labels = aug_fn(img.copy(), [l[:] for l in labels])

            if not aug_labels:
                continue

            stem = f"{img_path.stem}_aug{i:02d}_{aug_name}"
            cv2.imwrite(str(output_images / f"{stem}.jpg"), aug_img)
            save_labels(aug_labels, output_labels / f"{stem}.txt")


def main():
    parser = argparse.ArgumentParser(
        description="Augment annotated FOD dataset."
    )
    parser.add_argument("--input", required=True, help="Path to annotated/ directory")
    parser.add_argument("--output", required=True, help="Path to write augmented dataset")
    parser.add_argument("--factor", type=int, default=5, help="Augmentation multiplier (default: 5)")
    parser.add_argument(
        "--split", default=None,
        choices=["train", "val", "test"],
        help="Augment a single split only (default: train only)"
    )
    parser.add_argument(
        "--clean-bg-dir", default=None,
        help="Directory of clean tarmac images for copy-paste augmentation"
    )
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    # Gather clean backgrounds for copy-paste
    clean_backgrounds = []
    if args.clean_bg_dir:
        bg_dir = Path(args.clean_bg_dir)
        clean_backgrounds = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png"))
        print(f"Clean backgrounds : {len(clean_backgrounds)} images from {bg_dir}")

    splits_to_augment = [args.split] if args.split else ["train"]

    for split in splits_to_augment:
        print(f"\n── Split: {split} ──")
        augment_split(
            input_images=input_root / "images" / split,
            input_labels=input_root / "labels" / split,
            output_images=output_root / "images" / split,
            output_labels=output_root / "labels" / split,
            factor=args.factor,
            clean_backgrounds=clean_backgrounds,
        )

    print("\nAugmentation complete.")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
