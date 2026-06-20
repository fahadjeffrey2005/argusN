"""
finetune_v2.py — Fine-tune YOLOfinetune from best.pt on expanded dataset.

Continues from epoch-60 best.pt with new concrete runway surface data merged in.

Run from argusN/yolofinetune/:
    python scripts/finetune_v2.py
    python scripts/finetune_v2.py --epochs 60 --batch 32 --device cuda:0
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/yolo/finetuned/best.pt")
    parser.add_argument("--data",    default="../data/yolo_dataset/dataset.yaml")
    parser.add_argument("--epochs",  type=int,   default=40)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--device",  default="cuda")
    parser.add_argument("--lr",      type=float, default=0.0001)
    args = parser.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data}")

    print(f"\n{'='*55}")
    print(f"  YOLOfinetune v2 — Fine-tuning from checkpoint")
    print(f"{'='*55}")
    print(f"  Checkpoint : {weights}")
    print(f"  Dataset    : {data}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch      : {args.batch}")
    print(f"  Device     : {args.device}")
    print(f"  LR         : {args.lr}")
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

    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = Path("models/yolo/finetuned/best_v2.pt")
    best_dst.parent.mkdir(parents=True, exist_ok=True)

    if best_src.exists():
        shutil.copy(best_src, best_dst)
        print(f"\nBest weights saved → {best_dst}")
    else:
        print(f"\nWARNING: best.pt not found at {best_src}")

    print("Done. Run evaluate_fod.py to compare v2 vs original.")


if __name__ == "__main__":
    main()
