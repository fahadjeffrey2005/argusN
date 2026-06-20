"""
finetune_v2.py — Fine-tune YOLOfinetune from best.pt checkpoint on expanded dataset.
======================================================================================

Continues training from the existing epoch-60 best.pt checkpoint with the new
annotated data merged in (concrete runway surfaces, new FOD examples).

Key differences from original train_yolo.py:
  - Starts from best.pt (NOT base yolov8n.pt) — preserves all previous learning
  - Uses data/yolo_dataset/ (merged dataset with new images)
  - Lower LR (0.0001) — fine-tuning, not training from scratch
  - 40 epochs — enough to adapt to new surfaces without overfitting

Run from argusN/yolofinetune/ directory:
    python scripts/finetune_v2.py

    # Override defaults:
    python scripts/finetune_v2.py --epochs 60 --batch 32 --device cuda:0
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOfinetune v2 from best.pt checkpoint")
    parser.add_argument("--weights", default="models/yolo/finetuned/best.pt",
                        help="Starting checkpoint (default: models/yolo/finetuned/best.pt)")
    parser.add_argument("--data",    default="../data/yolo_dataset/dataset.yaml",
                        help="Dataset yaml (default: ../data/yolo_dataset/dataset.yaml)")
    parser.add_argument("--epochs",  type=int,   default=40)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--device",  default="cuda",
                        help="cuda / cuda:0 / cpu / mps")
    parser.add_argument("--lr",      type=float, default=0.0001,
                        help="Initial LR — keep low for fine-tuning (default 0.0001)")
    args = parser.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights}\n"
            "Rsync best.pt from Mac first:\n"
            "  rsync -avz <mac>:/Volumes/T72/argusN/yolofinetune/models/yolo/finetuned/best.pt "
            "./models/yolo/finetuned/best.pt"
        )

    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(
            f"Dataset yaml not found: {data}\n"
            "Rsync images from Mac first:\n"
            "  rsync -avz <mac>:/Volumes/T72/argusN/data/yolo_dataset/ ../data/yolo_dataset/"
        )

    print(f"\n{'='*55}")
    print(f"  YOLOfinetune v2 — Fine-tuning")
    print(f"{'='*55}")
    print(f"  Checkpoint : {weights}")
    print(f"  Dataset    : {data}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch      : {args.batch}")
    print(f"  Device     : {args.device}")
    print(f"  LR         : {args.lr}  (low — fine-tuning mode)")
    print(f"{'='*55}\n")

    model = YOLO(str(weights))

    results = model.train(
        data=str(data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        lr0=args.lr,
        lrf=0.01,
        project="models/yolo/runs",
        name="finetuned_v2",
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )

    # Copy best weights to standard location
    save_dir = Path(results.save_dir)
    best_src = save_dir / "weights" / "best.pt"
    best_dst = Path("models/yolo/finetuned/best_v2.pt")
    best_dst.parent.mkdir(parents=True, exist_ok=True)

    if best_src.exists():
        shutil.copy(best_src, best_dst)
        print(f"\nBest weights saved → {best_dst}")
    else:
        print(f"\nWARNING: best.pt not found at {best_src} — check models/yolo/runs/finetuned_v2/")

    print("\nDone. Run evaluate_fod.py to compare v2 vs original.")


if __name__ == "__main__":
    main()
