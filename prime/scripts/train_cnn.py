"""
PRIME — train_cnn.py
Trains the MobileNetV3-Small 4-channel CNN classifier on labeled crops.
Reads from data/crops/<class_name>/*.png
Saves best weights to models/cnn/prime_classifier.pth

Usage (from inside prime/ on Ubuntu):
    python scripts/train_cnn.py
    python scripts/train_cnn.py --crops-dir data/crops --epochs 30
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
        self.samples = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            class_dir = crops_root / class_name
            if not class_dir.exists():
                continue
            for p in sorted(class_dir.glob("*.png")):
                self.samples.append((p, class_id))

        if not self.samples:
            raise RuntimeError(
                f"No labeled crops in {crops_root}.\n"
                f"Run collect_crops.py then label_crops.py first."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img is None or img.ndim < 2:
            return torch.zeros(4, 128, 128), label

        # Ensure 4 channels
        if img.ndim == 2:
            img = np.stack([img, img, img, np.zeros_like(img)], axis=2)
        elif img.shape[2] == 3:
            img = np.concatenate([img, np.zeros((*img.shape[:2], 1), dtype=img.dtype)], axis=2)

        crop = img.astype(np.float32) / 255.0       # (H, W, 4)
        crop = np.transpose(crop, (2, 0, 1))         # (4, H, W)
        return torch.from_numpy(crop), label


def class_weights(dataset: CropDataset, device: str) -> torch.Tensor:
    counts = [0] * len(CLASS_NAMES)
    for _, lbl in dataset.samples:
        counts[lbl] += 1
    total = sum(counts)
    w = [total / (len(CLASS_NAMES) * max(c, 1)) for c in counts]
    return torch.tensor(w, dtype=torch.float).to(device)


def main():
    parser = argparse.ArgumentParser(description="Train PRIME CNN classifier")
    parser.add_argument("--config",    default="config/config.yaml")
    parser.add_argument("--crops-dir", default="data/crops")
    parser.add_argument("--epochs",    type=int, default=None)
    parser.add_argument("--resume",    action="store_true")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    logger = get_logger("train_cnn",
                        cfg.get("logging", "log_path", default="logs/prime.log"),
                        cfg.get("logging", "level", default="INFO"))

    device   = cfg.device
    epochs   = args.epochs or cfg.get("cnn", "epochs", default=30)
    batch    = cfg.get("cnn", "batch_size", default=32)
    lr       = cfg.get("cnn", "learning_rate", default=0.0005)
    patience = cfg.get("cnn", "early_stopping_patience", default=5)
    out_path = cfg.get("cnn", "model_path", default="models/cnn/prime_classifier.pth")

    # ── Dataset ───────────────────────────────────────────
    dataset = CropDataset(Path(args.crops_dir))
    logger.info(f"Dataset: {len(dataset)} crops")
    for cid, cname in enumerate(CLASS_NAMES):
        n = sum(1 for _, l in dataset.samples if l == cid)
        logger.info(f"  {cname}: {n}")

    val_n   = max(1, int(len(dataset) * 0.2))
    train_n = len(dataset) - val_n
    train_set, val_set = random_split(dataset, [train_n, val_n])

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────
    clf = CNNClassifier(cfg)
    if not args.resume:
        # Build fresh — don't load potentially absent saved weights
        clf.model = clf._build_model().to(device)
    clf.set_train_mode()

    # ── Training ──────────────────────────────────────────
    cw        = class_weights(dataset, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.Adam(clf.model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val  = float("inf")
    no_improve = 0
    history   = []

    logger.info(f"Training for up to {epochs} epochs on {device}")

    for epoch in range(1, epochs + 1):
        # Train
        clf.set_train_mode()
        t_loss = t_correct = 0
        for crops, labels in tqdm(train_loader, desc=f"Ep {epoch:03d} train", leave=False):
            crops, labels = crops.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = clf.model(crops)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * crops.size(0)
            t_correct += (logits.argmax(1) == labels).sum().item()

        t_loss /= train_n
        t_acc   = t_correct / train_n

        # Validate
        clf.set_eval_mode()
        v_loss = v_correct = 0
        with torch.no_grad():
            for crops, labels in val_loader:
                crops, labels = crops.to(device), labels.to(device)
                logits = clf.model(crops)
                v_loss    += criterion(logits, labels).item() * crops.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()

        v_loss /= val_n
        v_acc   = v_correct / val_n
        scheduler.step(v_loss)

        logger.info(
            f"Ep {epoch:03d}/{epochs} | "
            f"train loss={t_loss:.4f} acc={t_acc:.3f} | "
            f"val loss={v_loss:.4f} acc={v_acc:.3f}"
        )
        history.append({"epoch": epoch, "train_loss": round(t_loss,4),
                         "train_acc": round(t_acc,4), "val_loss": round(v_loss,4),
                         "val_acc": round(v_acc,4)})

        if v_loss < best_val:
            best_val   = v_loss
            no_improve = 0
            clf.save_weights(out_path)
            logger.info(f"  ✓ Best saved (val_loss={v_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    hist_path = Path("logs/train_history.json")
    hist_path.parent.mkdir(exist_ok=True)
    hist_path.write_text(json.dumps(history, indent=2))

    logger.info(f"\nTraining complete.")
    logger.info(f"  Best val_loss : {best_val:.4f}")
    logger.info(f"  Weights       : {out_path}")
    logger.info(f"  History       : {hist_path}")


if __name__ == "__main__":
    main()
