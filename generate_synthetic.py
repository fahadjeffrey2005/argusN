"""
generate_synthetic.py
─────────────────────
Generates synthetic paste-composite training images from the Roboflow dataset.

What it does:
  1. Reads annotated images from Roboflow dataset
  2. Extracts FOD object crops (nails, wires, bolts, etc.) using the bounding box labels
  3. Pastes them onto random background patches from clean runway images
  4. Applies augmentations: lighting, blur, noise, perspective, color jitter
  5. Writes new image+label pairs to data/synthetic/

This is the same technique Siddhart used — multiplying 200 annotated images into
thousands of training examples. We do the same with our 125 Roboflow images.

Run from argusN/ on Ubuntu:
    python generate_synthetic.py --data data/roboflow --out data/synthetic --count 1500

Args:
    --data   Path to Roboflow download (contains train/valid/test)
    --out    Output directory for synthetic data
    --count  Total synthetic images to generate (default 1500)
    --seed   Random seed for reproducibility (default 42)
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


# ── Config ───────────────────────────────────────────────────
MIN_TEMPLATE_PX = 8      # Skip FOD crops smaller than this (too tiny to be useful)
MAX_TEMPLATE_PX = 200    # Skip huge crops (likely annotation errors)
PASTES_PER_BG   = 3      # FOD objects pasted per background image
MARGIN_FRAC     = 0.05   # Don't paste within 5% of image edges


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",  default="data/roboflow", help="Roboflow dataset root")
    p.add_argument("--out",   default="data/synthetic", help="Output directory")
    p.add_argument("--count", type=int, default=1500,  help="Number of synthetic images")
    p.add_argument("--seed",  type=int, default=42)
    return p.parse_args()


# ── Augmentation helpers ─────────────────────────────────────

def aug_brightness(img, factor_range=(0.4, 1.6)):
    f = random.uniform(*factor_range)
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)

def aug_blur(img, max_ksize=5):
    k = random.choice([1, 1, 1, 3, 5])
    if k > 1:
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img

def aug_noise(img, std_range=(0, 15)):
    std = random.uniform(*std_range)
    noise = np.random.randn(*img.shape).astype(np.float32) * std
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def aug_fog(img, intensity_range=(0, 0.35)):
    intensity = random.uniform(*intensity_range)
    fog_layer = np.full_like(img, 200, dtype=np.float32)
    return np.clip(img.astype(np.float32) * (1 - intensity) + fog_layer * intensity,
                   0, 255).astype(np.uint8)

def aug_heatshimmer(img):
    """Simulate heat shimmer by slight wavy distortion."""
    h, w = img.shape[:2]
    map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)
    amplitude = random.uniform(0, 2.0)
    freq = random.uniform(0.02, 0.08)
    map_x += amplitude * np.sin(2 * np.pi * freq * map_y)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)

def aug_perspective(img, max_shift=0.04):
    h, w = img.shape[:2]
    s = max_shift
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([
        [random.uniform(0, s*w), random.uniform(0, s*h)],
        [w - random.uniform(0, s*w), random.uniform(0, s*h)],
        [w - random.uniform(0, s*w), h - random.uniform(0, s*h)],
        [random.uniform(0, s*w), h - random.uniform(0, s*h)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def augment_template(crop):
    """Augment the FOD template before pasting."""
    # Random flip
    if random.random() < 0.5:
        crop = cv2.flip(crop, 1)
    if random.random() < 0.3:
        crop = cv2.flip(crop, 0)
    # Random rotation
    angle = random.uniform(-180, 180)
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    crop = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    # Scale slightly
    scale = random.uniform(0.7, 1.3)
    nh, nw = max(4, int(h*scale)), max(4, int(w*scale))
    crop = cv2.resize(crop, (nw, nh))
    # Brightness
    crop = aug_brightness(crop, (0.5, 1.5))
    # Blur
    crop = aug_blur(crop, 3)
    return crop

def augment_background(bg):
    """Augment the background image."""
    choices = [aug_brightness, aug_noise, aug_fog, aug_heatshimmer, aug_perspective]
    random.shuffle(choices)
    for fn in choices[:random.randint(1, 3)]:
        bg = fn(bg)
    return bg


# ── Core synthesis ───────────────────────────────────────────

def extract_templates(data_root: Path, min_px: int, max_px: int):
    """
    Extract (class_id, crop_image) tuples from all annotated images.
    Returns list of (cls_id, np.ndarray) pairs.
    """
    templates = []
    img_dir = data_root / "train" / "images"
    lbl_dir = data_root / "train" / "labels"

    if not img_dir.exists():
        print(f"  WARNING: {img_dir} not found, trying root...")
        img_dir = data_root / "images"
        lbl_dir = data_root / "labels"

    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"  Found {len(img_files)} training images for template extraction")

    for img_path in img_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(lbl_path) as f:
            lines = f.read().strip().split("\n")

        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop_h = y2 - y1
            crop_w = x2 - x1

            if crop_h < min_px or crop_w < min_px:
                continue
            if crop_h > max_px or crop_w > max_px:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            templates.append((cls_id, crop.copy()))

    print(f"  Extracted {len(templates)} FOD templates")
    return templates


def collect_backgrounds(data_root: Path, extra_bg_dirs: list = None):
    """Collect background image paths — use annotated training images as backgrounds too."""
    bgs = []
    for split in ["train", "valid"]:
        img_dir = data_root / split / "images"
        if img_dir.exists():
            bgs.extend(img_dir.glob("*.jpg"))
            bgs.extend(img_dir.glob("*.png"))

    if extra_bg_dirs:
        for d in extra_bg_dirs:
            p = Path(d)
            if p.exists():
                bgs.extend(p.glob("*.jpg"))
                bgs.extend(p.glob("*.png"))

    print(f"  Found {len(bgs)} background images")
    return bgs


def paste_fod_on_bg(bg_img, templates, cls_names, n_pastes=3):
    """
    Paste n_pastes FOD templates onto bg_img.
    Returns: (result_image, list_of_yolo_labels)
    """
    h, w = bg_img.shape[:2]
    result = bg_img.copy()
    labels = []

    margin_x = int(MARGIN_FRAC * w)
    margin_y = int(MARGIN_FRAC * h)

    for _ in range(n_pastes):
        cls_id, crop = random.choice(templates)
        crop = augment_template(crop)

        ch, cw = crop.shape[:2]
        if ch <= 0 or cw <= 0:
            continue

        # Random placement (avoid edges)
        max_x = w - cw - margin_x
        max_y = h - ch - margin_y
        if max_x <= margin_x or max_y <= margin_y:
            continue

        px = random.randint(margin_x, max_x)
        py = random.randint(margin_y, max_y)

        # Simple paste (alpha blend with slight edge feathering)
        roi = result[py:py+ch, px:px+cw]
        blend = random.uniform(0.75, 1.0)
        result[py:py+ch, px:px+cw] = np.clip(
            roi.astype(np.float32) * (1 - blend) + crop.astype(np.float32) * blend,
            0, 255
        ).astype(np.uint8)

        # YOLO label (normalized)
        cx_n = (px + cw/2) / w
        cy_n = (py + ch/2) / h
        bw_n = cw / w
        bh_n = ch / h
        labels.append(f"{cls_id} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

    return result, labels


# ── Main ─────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data)
    out_root  = Path(args.out)

    print(f"\n{'='*55}")
    print(f"  Hawkeye Synthetic Data Generator")
    print(f"{'='*55}")
    print(f"  Source  : {data_root}")
    print(f"  Output  : {out_root}")
    print(f"  Target  : {args.count} synthetic images")
    print(f"{'='*55}\n")

    # Load class names from data.yaml
    import yaml as pyyaml
    yaml_path = data_root / "data.yaml"
    cls_names = {}
    if yaml_path.exists():
        with open(yaml_path) as f:
            cfg = pyyaml.safe_load(f)
        names = cfg.get("names", [])
        if isinstance(names, list):
            cls_names = {i: n for i, n in enumerate(names)}
        elif isinstance(names, dict):
            cls_names = names
        print(f"  Classes: {cls_names}\n")

    # Setup output dirs
    for split in ["train", "valid"]:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    # Extract templates
    print("[1] Extracting FOD templates...")
    templates = extract_templates(data_root, MIN_TEMPLATE_PX, MAX_TEMPLATE_PX)

    if len(templates) == 0:
        print("  ERROR: No templates extracted. Check that data/roboflow/train/images and labels exist.")
        return

    # Collect backgrounds
    print("\n[2] Collecting backgrounds...")
    bgs = collect_backgrounds(data_root)

    if len(bgs) == 0:
        print("  ERROR: No background images found.")
        return

    # Generate
    print(f"\n[3] Generating {args.count} synthetic images...")

    n_train = int(args.count * 0.85)
    n_valid = args.count - n_train

    total = 0
    for split, n_target in [("train", n_train), ("valid", n_valid)]:
        img_out = out_root / split / "images"
        lbl_out = out_root / split / "labels"

        for i in range(n_target):
            bg_path = random.choice(bgs)
            bg = cv2.imread(str(bg_path))
            if bg is None:
                continue

            # Resize to 640x640
            bg = cv2.resize(bg, (640, 640))

            # Augment background
            bg = augment_background(bg)

            # Paste FODs
            n_pastes = random.randint(1, PASTES_PER_BG)
            result, labels = paste_fod_on_bg(bg, templates, cls_names, n_pastes)

            if not labels:
                continue

            fname = f"synth_{split}_{i:05d}"
            cv2.imwrite(str(img_out / f"{fname}.jpg"), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (lbl_out / f"{fname}.txt").write_text("\n".join(labels))

            total += 1
            if total % 100 == 0 or total == args.count:
                print(f"    {total}/{args.count} generated...", flush=True)

    print(f"\n  Done! Generated {total} synthetic images")
    print(f"  Train : {n_train}")
    print(f"  Valid : {n_valid}")

    # Write a merged dataset.yaml combining real + synthetic
    merged_yaml = out_root / "data.yaml"
    nc = len(cls_names)
    names_list = [cls_names[i] for i in sorted(cls_names.keys())]
    yaml_content = f"""# Hawkeye Synthetic Dataset
# Generated by generate_synthetic.py
# Use this yaml for retraining

path: {out_root.resolve()}
train: train/images
val:   valid/images

nc: {nc}
names: {names_list}
"""
    merged_yaml.write_text(yaml_content)
    print(f"\n  Dataset YAML: {merged_yaml}")

    print(f"\n{'='*55}")
    print(f"  NEXT STEP — Merge with real data and retrain:")
    print(f"")
    print(f"  python retrain.py \\")
    print(f"    --real    {data_root} \\")
    print(f"    --synth   {out_root} \\")
    print(f"    --weights yolofinetune/models/yolo/finetuned/best.pt")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
