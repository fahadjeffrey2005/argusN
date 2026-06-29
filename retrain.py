"""
retrain.py — Retrain best.pt on real + synthetic data combined.
─────────────────────────────────────────────────────────────────
Strategy:
  1. Merge real Roboflow data + synthetic generated data into one dataset
  2. Fine-tune from best.pt (continue learning, don't reset weights)
  3. Use lower LR (fine-tuning, not from scratch)
  4. Use NWD loss if supported by installed Ultralytics version
  5. Cosine LR schedule + longer warmup for stable tiny-object training

Run from argusN/ on Ubuntu:
    python retrain.py --real data/roboflow --synth data/synthetic

Optional overrides:
    python retrain.py --real data/roboflow --synth data/synthetic \\
        --epochs 80 --batch 16 --imgsz 640 --device cuda:0

Outputs:
    yolofinetune/models/yolo/runs/retrain_v1/weights/best.pt
    (copy this over the old best.pt once validated)
"""

import argparse
import shutil
import sys
from pathlib import Path
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real",    default="data/roboflow",  help="Roboflow dataset root")
    p.add_argument("--synth",   default="data/synthetic", help="Synthetic dataset root")
    p.add_argument("--weights", default="yolofinetune/models/yolo/finetuned/best.pt")
    p.add_argument("--epochs",  type=int,   default=80)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--device",  default="cuda")
    p.add_argument("--lr",      type=float, default=0.0005,  help="Initial LR (low for fine-tuning)")
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--name",    default="retrain_v1")
    p.add_argument("--patience",type=int,   default=30,   help="Early stopping patience")
    return p.parse_args()


def merge_datasets(real_root: Path, synth_root: Path, merged_root: Path):
    """
    Merge real + synthetic datasets into a single directory.
    Copies images and labels from both sources, prefixed to avoid name collisions.
    """
    print(f"\n[1] Merging datasets -> {merged_root}")
    merged_root.mkdir(parents=True, exist_ok=True)

    # Read class names from real dataset yaml
    real_yaml = real_root / "data.yaml"
    if not real_yaml.exists():
        real_yaml = real_root / "dataset.yaml"
    with open(real_yaml) as f:
        real_cfg = yaml.safe_load(f)

    nc    = real_cfg.get("nc", real_cfg.get("num_classes", 1))
    names = real_cfg.get("names", ["fod"])

    counts = {}
    for split in ["train", "valid"]:
        out_img = merged_root / split / "images"
        out_lbl = merged_root / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        n = 0
        # Real data
        for src_name, prefix in [(real_root, "real"), (synth_root, "synth")]:
            src_img = src_name / split / "images"
            src_lbl = src_name / split / "labels"
            if not src_img.exists():
                continue
            for img_path in list(src_img.glob("*.jpg")) + list(src_img.glob("*.png")):
                lbl_path = src_lbl / (img_path.stem + ".txt")
                # For synthetic: only include if label exists and is non-empty
                if prefix == "synth":
                    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                        continue
                dst_stem = f"{prefix}_{img_path.stem}"
                shutil.copy2(img_path, out_img / f"{dst_stem}{img_path.suffix}")
                if lbl_path.exists():
                    shutil.copy2(lbl_path, out_lbl / f"{dst_stem}.txt")
                else:
                    # Empty label file (background image for real data)
                    (out_lbl / f"{dst_stem}.txt").write_text("")
                n += 1
        counts[split] = n
        print(f"    {split:6s}: {n} images")

    # Write merged data.yaml
    merged_yaml = {
        "path":  str(merged_root.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "nc":    nc,
        "names": names,
    }
    yaml_path = merged_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(merged_yaml, f, sort_keys=False)
    print(f"    YAML: {yaml_path}")
    print(f"    Classes ({nc}): {names}")

    return yaml_path, counts


def main():
    args = parse_args()

    real_root  = Path(args.real)
    synth_root = Path(args.synth)
    weights    = Path(args.weights)
    merged_root = Path("data/merged")

    print(f"\n{'='*60}")
    print(f"  Hawkeye Detector Retraining")
    print(f"{'='*60}")
    print(f"  Weights  : {weights}")
    print(f"  Real data: {real_root}")
    print(f"  Synth    : {synth_root}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Image sz : {args.imgsz}")
    print(f"  Device   : {args.device}")
    print(f"  LR       : {args.lr}")
    print(f"{'='*60}\n")

    if not weights.exists():
        print(f"ERROR: Weights not found: {weights}")
        sys.exit(1)

    if not real_root.exists():
        print(f"ERROR: Real data not found: {real_root}")
        print("Run: python download_roboflow.py --key YOUR_KEY")
        sys.exit(1)

    if not synth_root.exists():
        print(f"WARNING: Synthetic data not found at {synth_root}")
        print("Training on real data only. Run generate_synthetic.py first for best results.")
        # Fall back to real data only
        merged_root = real_root
        yaml_path = real_root / "data.yaml"
        if not yaml_path.exists():
            yaml_path = real_root / "dataset.yaml"
        counts = {}
    else:
        yaml_path, counts = merge_datasets(real_root, synth_root, merged_root)

    # ── Load model and train ─────────────────────────────────
    print(f"\n[2] Loading model from {weights}...")
    from ultralytics import YOLO
    model = YOLO(str(weights))

    print(f"\n[3] Starting training...")
    print(f"    Dataset YAML : {yaml_path}")
    print(f"    Output       : yolofinetune/models/yolo/runs/{args.name}/\n")

    # Check if NWD loss is available in this Ultralytics version
    try:
        # NWD is available in some community forks / newer versions
        # Standard Ultralytics uses CIoU. We use box_loss_type if available.
        import ultralytics
        ul_version = ultralytics.__version__
        print(f"    Ultralytics version: {ul_version}")
    except:
        pass

    train_kwargs = dict(
        data      = str(yaml_path),
        epochs    = args.epochs,
        batch     = args.batch,
        imgsz     = args.imgsz,
        device    = args.device,
        lr0       = args.lr,
        lrf       = 0.01,            # Final LR = lr0 * lrf
        warmup_epochs = 5,           # Longer warmup for fine-tuning stability
        cos_lr    = True,            # Cosine LR schedule
        patience  = args.patience,   # Early stop if no improvement
        workers   = args.workers,
        project   = "yolofinetune/models/yolo/runs",
        name      = args.name,
        exist_ok  = True,
        save      = True,
        val       = True,
        plots     = True,
        # Augmentation — moderate (our data already augmented synthetically)
        hsv_h     = 0.015,
        hsv_s     = 0.5,
        hsv_v     = 0.4,
        degrees   = 10.0,
        translate = 0.1,
        scale     = 0.5,
        shear     = 2.0,
        flipud    = 0.1,
        fliplr    = 0.5,
        mosaic    = 0.8,
        mixup     = 0.1,
        copy_paste= 0.1,
        # Tiny object improvements
        overlap_mask = False,
    )

    # Try NWD loss (only in some Ultralytics forks)
    # If it fails, fall back silently to standard training
    nwd_available = False
    try:
        results = model.train(bbox_loss="nwd", **train_kwargs)
        nwd_available = True
        print("\n    NWD loss active (better gradient signal for tiny objects)")
    except TypeError:
        print("\n    NWD loss not available in this Ultralytics version")
        print("    Falling back to standard CIoU loss")
        print("    To enable NWD: pip install ultralytics-nwd or use a fork that supports it")
        results = model.train(**train_kwargs)

    # ── Results summary ──────────────────────────────────────
    best_new = Path(f"yolofinetune/models/yolo/runs/{args.name}/weights/best.pt")
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  New best.pt: {best_new}")
    if best_new.exists():
        size_mb = best_new.stat().st_size / 1e6
        print(f"  Size: {size_mb:.1f} MB")
    print(f"  NWD loss used: {nwd_available}")
    print(f"\n  NEXT STEPS:")
    print(f"  1. Run compare_models.py to benchmark old vs new")
    print(f"  2. If improved: cp {best_new} yolofinetune/models/yolo/finetuned/best.pt")
    print(f"  3. Run ENHANCED pipeline on fod1.mp4 and clean1.mp4 with new weights")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
