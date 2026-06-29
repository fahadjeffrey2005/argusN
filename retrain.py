"""
retrain.py — Two-mode retraining script.
─────────────────────────────────────────────────────────────────
MODE 1 — Quick fine-tune (default):
    Fine-tune current best.pt (YOLOv8n) on Roboflow + synthetic data.
    Faster. Shows improvement quickly. Still YOLOv8n architecture.

    python retrain.py --real data/roboflow --synth data/synthetic

MODE 2 — Full upgrade to YOLOv11s (recommended):
    Start from COCO-pretrained YOLOv11s. Train on full dataset.
    Better architecture, 3x more parameters, better tiny-object capacity.
    Takes longer but this is the proper fix.

    python retrain.py --real data/roboflow --synth data/synthetic --upgrade

Outputs:
    yolofinetune/models/yolo/runs/retrain_v1/weights/best.pt  (fine-tune)
    yolofinetune/models/yolo/runs/upgrade_v11s/weights/best.pt (upgrade)

Once validated, copy the new best.pt over the old one:
    cp yolofinetune/models/yolo/runs/retrain_v1/weights/best.pt \\
       yolofinetune/models/yolo/finetuned/best.pt
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
    p.add_argument("--upgrade", action="store_true",
                   help="Upgrade to YOLOv11s instead of fine-tuning current best.pt")
    p.add_argument("--epochs",  type=int,   default=80)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--device",  default="cuda")
    p.add_argument("--lr",      type=float, default=None,
                   help="Initial LR. Default: 0.0005 for fine-tune, 0.01 for upgrade")
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--name",    default=None)
    p.add_argument("--patience",type=int,   default=30)
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
    merged_root = Path("data/merged")

    # ── Mode selection ────────────────────────────────────────
    if args.upgrade:
        # MODE 2: Upgrade to YOLOv11s — train from COCO pretrained weights
        # YOLOv11s: ~9M params, 3x more capacity than YOLOv8n
        # Better feature extraction, better tiny object handling
        model_source = "yolo11s.pt"   # Ultralytics auto-downloads pretrained COCO weights
        run_name     = args.name or "upgrade_v11s"
        lr0          = args.lr if args.lr else 0.01     # Standard LR for training from COCO pretrained
        warmup       = 3
        mode_label   = "UPGRADE (YOLOv11s from COCO pretrained)"
    else:
        # MODE 1: Fine-tune current best.pt (YOLOv8n)
        model_source = args.weights
        run_name     = args.name or "retrain_v1"
        lr0          = args.lr if args.lr else 0.0005   # Low LR for fine-tuning
        warmup       = 5
        mode_label   = f"FINE-TUNE ({args.weights})"

        if not Path(model_source).exists():
            print(f"ERROR: Weights not found: {model_source}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Hawkeye Detector Retraining")
    print(f"{'='*60}")
    print(f"  Mode     : {mode_label}")
    print(f"  Real data: {real_root}")
    print(f"  Synth    : {synth_root}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Image sz : {args.imgsz}")
    print(f"  Device   : {args.device}")
    print(f"  LR0      : {lr0}")
    print(f"  Run name : {run_name}")
    print(f"{'='*60}\n")

    if not real_root.exists():
        print(f"ERROR: Real data not found: {real_root}")
        print("Run: python download_roboflow.py --key YOUR_KEY")
        sys.exit(1)

    if not synth_root.exists():
        print(f"WARNING: Synthetic data not found at {synth_root}")
        print("Training on real data only. Run generate_synthetic.py first for best results.")
        merged_root = real_root
        yaml_path = real_root / "data.yaml"
        if not yaml_path.exists():
            yaml_path = real_root / "dataset.yaml"
    else:
        yaml_path, _ = merge_datasets(real_root, synth_root, merged_root)

    # ── Load model and train ─────────────────────────────────
    print(f"\n[2] Loading model: {model_source}")
    if args.upgrade:
        print("    Downloading YOLOv11s pretrained weights from Ultralytics (if not cached)...")
    from ultralytics import YOLO
    model = YOLO(model_source)

    # Log Ultralytics version
    try:
        import ultralytics
        print(f"    Ultralytics version: {ultralytics.__version__}")
    except Exception:
        pass

    print(f"\n[3] Starting training...")
    print(f"    Dataset YAML : {yaml_path}")
    print(f"    Output       : yolofinetune/models/yolo/runs/{run_name}/\n")

    train_kwargs = dict(
        data          = str(yaml_path),
        epochs        = args.epochs,
        batch         = args.batch,
        imgsz         = args.imgsz,
        device        = args.device,
        lr0           = lr0,
        lrf           = 0.01,
        warmup_epochs = warmup,
        cos_lr        = True,
        patience      = args.patience,
        workers       = args.workers,
        project       = "yolofinetune/models/yolo/runs",
        name          = run_name,
        exist_ok      = True,
        save          = True,
        val           = True,
        plots         = True,
        # Augmentation — moderate (our data already augmented synthetically)
        hsv_h         = 0.015,
        hsv_s         = 0.5,
        hsv_v         = 0.4,
        degrees       = 10.0,
        translate     = 0.1,
        scale         = 0.5,
        shear         = 2.0,
        flipud        = 0.1,
        fliplr        = 0.5,
        mosaic        = 0.8,
        mixup         = 0.1,
        copy_paste    = 0.1,
        overlap_mask  = False,
    )

    # Try NWD loss (available in some Ultralytics versions / forks)
    nwd_available = False
    try:
        results = model.train(bbox_loss="nwd", **train_kwargs)
        nwd_available = True
        print("\n    NWD loss active (better gradient signal for tiny objects)")
    except (TypeError, SyntaxError):
        print("\n    NWD loss not available in this Ultralytics version — using CIoU")
        results = model.train(**train_kwargs)

    # ── Results summary ──────────────────────────────────────
    best_new = Path(f"yolofinetune/models/yolo/runs/{run_name}/weights/best.pt")
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Mode         : {mode_label}")
    print(f"  New best.pt  : {best_new}")
    if best_new.exists():
        size_mb = best_new.stat().st_size / 1e6
        print(f"  Size         : {size_mb:.1f} MB")
    print(f"  NWD loss     : {nwd_available}")
    print(f"\n  NEXT STEPS:")
    print(f"  1. Run compare_models.py to benchmark old vs new")
    print(f"  2. If improved: cp {best_new} yolofinetune/models/yolo/finetuned/best.pt")
    print(f"  3. Run ENHANCED pipeline on fod1.mp4 and clean1.mp4 with new weights")
    if not args.upgrade:
        print(f"\n  For a full architecture upgrade, run:")
        print(f"  python retrain.py --real {real_root} --synth {synth_root} --upgrade")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
