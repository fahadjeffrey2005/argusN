"""
train_yolo.py — Fine-tune YOLOv8n on the runway FOD dataset.

Training strategy:
  - Load pretrained YOLOv8n weights
  - Freeze first 10 layers (backbone) — transfer low-level features
  - Train detection head on single-class FOD dataset
  - Save best weights to models/yolo/finetuned/best.pt

Run from yolofinetune/ root:
    python scripts/train_yolo.py

    # Override config values:
    python scripts/train_yolo.py --epochs 100 --batch 8 --device cpu
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add project root to path so src imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger


def verify_dataset(data_yaml: Path) -> bool:
    """Check that dataset.yaml exists and referenced directories are populated."""
    if not data_yaml.exists():
        return False
    with open(data_yaml) as f:
        ds = yaml.safe_load(f)
    train_path = Path(ds.get("train", ""))
    val_path = Path(ds.get("val", ""))
    train_imgs = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png")) if train_path.exists() else []
    val_imgs = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png")) if val_path.exists() else []
    print(f"Dataset  train : {len(train_imgs)} images")
    print(f"Dataset  val   : {len(val_imgs)} images")
    return len(train_imgs) > 0 and len(val_imgs) > 0


def train(cfg_path: str = "config/config.yaml", overrides: dict = None):
    from ultralytics import YOLO

    cfg = load_config(cfg_path)
    log = get_logger(
        "train",
        cfg.get("logging", "log_path", default="logs/yolofinetune.log"),
        cfg.get("logging", "level", default="INFO")
    )

    overrides = overrides or {}

    pretrained_path = overrides.get("pretrained_path") or cfg.get("yolo", "pretrained_path")
    output_path     = overrides.get("model_path") or cfg.get("yolo", "model_path")
    device          = overrides.get("device") or cfg.device
    epochs          = overrides.get("epochs") or cfg.get("yolo", "epochs", default=50)
    batch_size      = overrides.get("batch_size") or cfg.get("yolo", "batch_size", default=16)
    lr              = overrides.get("learning_rate") or cfg.get("yolo", "learning_rate", default=0.001)
    imgsz           = cfg.get("yolo", "input_size", default=640)
    freeze_layers   = cfg.get("yolo", "freeze_layers", default=10)

    data_yaml = Path("config/dataset.yaml")

    log.info("=" * 50)
    log.info("YOLOFINETUNE — Training")
    log.info("=" * 50)
    log.info(f"Pretrained weights : {pretrained_path}")
    log.info(f"Output model       : {output_path}")
    log.info(f"Device             : {device}")
    log.info(f"Epochs             : {epochs}")
    log.info(f"Batch size         : {batch_size}")
    log.info(f"Learning rate      : {lr}")
    log.info(f"Image size         : {imgsz}")
    log.info(f"Freeze layers      : {freeze_layers}")
    log.info(f"Dataset yaml       : {data_yaml}")

    if not Path(pretrained_path).exists():
        log.error(f"Pretrained weights not found: {pretrained_path}")
        log.error("Run setup.sh first or download yolov8n.pt manually.")
        sys.exit(1)

    if not verify_dataset(data_yaml):
        log.error(f"Dataset not ready: {data_yaml}")
        log.error("Run augment_dataset.py first, then check config/dataset.yaml paths.")
        sys.exit(1)

    # Load pretrained model
    log.info("Loading pretrained YOLOv8n...")
    model = YOLO(pretrained_path)

    # Freeze backbone layers
    log.info(f"Freezing first {freeze_layers} layers...")
    for i, (name, param) in enumerate(model.model.named_parameters()):
        if i < freeze_layers:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.model.parameters())
    log.info(f"Trainable params  : {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Train
    log.info("Starting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        lr0=lr,
        project="models/yolo/runs",
        name="finetuned",
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )

    # Copy best weights to expected path
    best_src = Path("models/yolo/runs/finetuned/weights/best.pt")
    best_dst = Path(output_path)
    best_dst.parent.mkdir(parents=True, exist_ok=True)

    if best_src.exists():
        import shutil
        shutil.copy(best_src, best_dst)
        log.info(f"Best weights saved → {best_dst}")
    else:
        log.warning(f"Expected best.pt not found at {best_src}")

    log.info("Training complete.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n for FOD detection.")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", default=None, help="Override device (cuda / cpu / mps)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    overrides = {}
    if args.epochs:  overrides["epochs"] = args.epochs
    if args.batch:   overrides["batch_size"] = args.batch
    if args.device:  overrides["device"] = args.device
    if args.lr:      overrides["learning_rate"] = args.lr

    train(cfg_path=args.config, overrides=overrides)


if __name__ == "__main__":
    main()
