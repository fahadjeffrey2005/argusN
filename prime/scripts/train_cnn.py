"""
PRIME — train_cnn.py
Trains the MobileNetV3-Small 4-channel CNN classifier.
Reads labeled crops from data/crops/<class_name>/*.png
Saves best weights to models/cnn/prime_classifier.pth

Usage:
    python scripts/train_cnn.py --config config/config.yaml
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.semantic.cnn_classifier import CNNClassifier

CLASS_NAMES = ["fod", "shadow", "runway_marking", "strobe_light", "clean_tarmac"]


class CropDataset(Dataset):
    def __init__(self, crops_root: Path):
        import cv2
        self.samples = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            class_dir = crops_root / class_name
            if not class_dir.exists():
                continue
            for crop_path in class_dir.glob("*.png"):
                self.samples.append((crop_path, class_id))

        if not self.samples:
            raise RuntimeError(
                f"No labeled crops found in {crops_root}. "
                f"Run collect_crops.py then label_crops.py first."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img is None or img.ndim < 3:
            # Fallback: return zeros
            return torch.zeros(4, 128, 128), label

        if img.shape[2] == 3:
            # Add zero flow channel if only 3-channel saved
            zero_flow = np.zeros((*img.shape[:2], 1), dtype=img.dtype)
            img = np.concatenate([img, zero_flow], axis=2)

        # (H, W, 4) → (4, H, W), normalise
        crop = img.astype(np.float32) / 255.0
        crop = np.transpose(crop, (2, 0, 1))
        return torch.from_numpy(crop), label


def get_class_weights(dataset: CropDataset, device: str) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = [0] * len(CLASS_NAMES)
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (len(CLASS_NAMES) * max(c, 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float).to(device)


def main():
    parser = argparse.ArgumentParser(description="Train PRIME CNN classifier")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--crops-dir", default="data/crops", help="Root of labeled crops")
    parser.add_argument("--resume", action="store_true", help="Resume from existing weights")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger(
        "train_cnn",
        cfg.get("logging", "log_path", default="logs/prime.log"),
        cfg.get("logging", "level", default="INFO")
    )

    device = cfg.device
    epochs = cfg.get("cnn", "epochs", default=30)
    batch_size = cfg.get("cnn", "batch_size", default=32)
    lr = cfg.get("cnn", "learning_rate", default=0.0005)
    patience = cfg.get("cnn", "early_stopping_patience", default=5)
    model_path = cfg.get("cnn", "model_path", default="models/cnn/prime_classifier.pth")

    # ── Dataset ────────────────────────────────────────────────────
    crops_root = Path(args.crops_dir)
    dataset = CropDataset(crops_root)

    logger.info(f"Dataset: {len(dataset)} crops across {len(CLASS_NAMES)} classes")
    for class_id, class_name in enumerate(CLASS_NAMES):
        count = sum(1 for _, lbl in dataset.samples if lbl == class_id)
        logger.info(f"  {class_name}: {count}")

    # 80/20 train/val split
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # ── Model ──────────────────────────────────────────────────────
    # Build directly (don't load trained weights if starting fresh)
    classifier = CNNClassifier(cfg)
    if args.resume and Path(model_path).exists():
        logger.info(f"Resuming from {model_path}")
    else:
        # Re-build fresh model (without loading potentially absent weights)
        classifier.model = classifier._build_model().to(device)

    classifier.set_train_mode()

    # ── Loss, optimiser ────────────────────────────────────────────
    class_weights = get_class_weights(dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(classifier.model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # ── Training loop ──────────────────────────────────────────────
    best_val_loss = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        # Train
        classifier.set_train_mode()
        train_loss = 0.0
        train_correct = 0

        for crops, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            crops = crops.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = classifier.model(crops)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * crops.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()

        train_loss /= train_size
        train_acc = train_correct / train_size

        # Validate
        classifier.set_eval_mode()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for crops, labels in val_loader:
                crops = crops.to(device)
                labels = labels.to(device)
                logits = classifier.model(crops)
                loss = criterion(logits, labels)
                val_loss += loss.item() * crops.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()

        val_loss /= val_size
        val_acc = val_correct / val_size

        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4)
        })

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            classifier.save_weights(model_path)
            logger.info(f"  → Best model saved (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Save training history
    history_path = Path("logs/train_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    logger.info(f"Weights → {model_path}")
    logger.info(f"History → {history_path}")


if __name__ == "__main__":
    main()
