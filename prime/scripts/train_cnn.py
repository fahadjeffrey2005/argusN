"""
PRIME — train_cnn.py
Train MobileNetV3-Small CNN classifier on labeled 4-channel crops.

Architecture decisions (locked):
  - Backbone frozen except first conv layer + classifier head
  - First conv learns the frame-diff 4th channel; pretrained RGB weights kept
  - Weighted cross-entropy loss with inverse class frequency weights
  - 4-channel augmentation applied to all classes (heavier noise on FOD)
  - 80/20 stratified train/val split

Dataset structure (data/crops/labeled_crops/):
    fod/
    shadow/
    runway_marking/
    strobe_light/
    clean_tarmac/

Usage (from inside prime/ on Ubuntu):
    python scripts/train_cnn.py
    python scripts/train_cnn.py --data data/crops/labeled_crops --epochs 40
    python scripts/train_cnn.py --data data/crops/labeled_crops --input-size 224
"""

import argparse
import sys
import random
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

CLASS_NAMES = ["fod", "shadow", "runway_marking", "strobe_light", "clean_tarmac"]
FOD_CLASS_ID = 0


# ── Dataset ───────────────────────────────────────────────────────────────────

class CropDataset(Dataset):
    """
    Loads 4-channel PNG crops from a class-folder directory structure.
    Applies 4-channel augmentation (geometric + photometric).
    """

    def __init__(self, samples: list, input_size: int, augment: bool = False):
        self.samples    = samples
        self.input_size = input_size
        self.augment    = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_id = self.samples[idx]

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img is None:
            return torch.zeros(4, self.input_size, self.input_size), class_id

        # Ensure exactly 4 channels
        if img.ndim == 2:
            img = np.stack([img, img, img, np.zeros_like(img)], axis=2)
        elif img.shape[2] == 3:
            img = np.concatenate(
                [img, np.zeros((*img.shape[:2], 1), dtype=img.dtype)], axis=2
            )
        elif img.shape[2] > 4:
            img = img[:, :, :4]

        # Resize if needed
        if img.shape[0] != self.input_size or img.shape[1] != self.input_size:
            img = cv2.resize(img, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_LINEAR)

        # (H, W, 4) → float32 [0, 1], channel-first
        tensor = torch.from_numpy(
            img.astype(np.float32) / 255.0
        ).permute(2, 0, 1)   # (4, H, W)

        if self.augment:
            tensor = self._augment(tensor, class_id)

        return tensor, class_id

    @staticmethod
    def _augment(tensor: torch.Tensor, class_id: int) -> torch.Tensor:
        """
        4-channel augmentation.
        Geometric ops applied to all channels identically.
        Photometric ops applied only to BGR channels (0-2).
        FOD class gets heavier noise to improve cross-runway recall.
        """
        # Geometric
        if random.random() > 0.5:
            tensor = torch.flip(tensor, dims=[2])   # horizontal flip
        if random.random() > 0.5:
            tensor = torch.flip(tensor, dims=[1])   # vertical flip
        k = random.randint(0, 3)
        if k > 0:
            tensor = torch.rot90(tensor, k, dims=[1, 2])

        # Brightness jitter
        brightness = 1.0 + (random.random() - 0.5) * 0.4
        tensor[:3] = (tensor[:3] * brightness).clamp(0.0, 1.0)

        # Contrast jitter
        contrast = 1.0 + (random.random() - 0.5) * 0.3
        mean = tensor[:3].mean()
        tensor[:3] = ((tensor[:3] - mean) * contrast + mean).clamp(0.0, 1.0)

        # Saturation jitter
        if random.random() > 0.5:
            sat  = 1.0 + (random.random() - 0.5) * 0.3
            gray = tensor[:3].mean(dim=0, keepdim=True)
            tensor[:3] = (gray + (tensor[:3] - gray) * sat).clamp(0.0, 1.0)

        # Gaussian noise — heavier for FOD to improve cross-runway recall
        noise_std = 0.06 if class_id == FOD_CLASS_ID else 0.025
        noise = torch.randn_like(tensor[:3]) * noise_std
        tensor[:3] = (tensor[:3] + noise).clamp(0.0, 1.0)

        return tensor


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, device: str) -> nn.Module:
    """
    MobileNetV3-Small with 4-channel input.
    Frozen backbone — only first conv + classifier head are trainable.
    """
    try:
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    except Exception:
        from torchvision.models import mobilenet_v3_small
        model = mobilenet_v3_small(pretrained=True)

    # Freeze entire backbone
    for param in model.parameters():
        param.requires_grad = False

    # First conv: 3 → 4 channels
    original_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = original_conv.weight.data
        nn.init.constant_(new_conv.weight[:, 3:, :, :], 0.0)
        if original_conv.bias is not None:
            new_conv.bias = nn.Parameter(original_conv.bias.data.clone())
    model.features[0][0] = new_conv

    # Unfreeze first conv (must learn frame-diff channel)
    for param in model.features[0][0].parameters():
        param.requires_grad = True

    # Replace + unfreeze classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params : {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")

    return model.to(device)


# ── Data utilities ─────────────────────────────────────────────────────────────

def load_samples(data_dir: Path) -> list:
    samples = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"  WARNING: class folder not found — {class_dir}")
            continue
        paths = sorted(class_dir.glob("*.png"))
        samples.extend([(p, class_id) for p in paths])
        print(f"  {class_name:<22} {len(paths):5d} crops")
    return samples


def train_val_split(samples: list, val_frac: float = 0.20, seed: int = 42):
    """Stratified 80/20 split within each class."""
    rng = random.Random(seed)
    by_class = {i: [] for i in range(len(CLASS_NAMES))}
    for path, class_id in samples:
        by_class[class_id].append((path, class_id))

    train, val = [], []
    for class_id, items in by_class.items():
        rng.shuffle(items)
        n_val  = max(1, int(len(items) * val_frac))
        val   += items[:n_val]
        train += items[n_val:]
    return train, val


def compute_class_weights(samples: list, num_classes: int,
                          device: str) -> torch.Tensor:
    counts = [0] * num_classes
    for _, class_id in samples:
        counts[class_id] += 1
    total   = sum(counts)
    weights = [total / max(c, 1) for c in counts]
    mean_w  = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
    print("  Class weights:")
    for i, (name, w) in enumerate(zip(CLASS_NAMES, weights)):
        print(f"    {name:<22} count={counts[i]:4d}  weight={w:.3f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = correct = n = 0

    with torch.set_grad_enabled(train):
        for tensors, labels in loader:
            tensors = tensors.to(device)
            labels  = labels.to(device)

            logits = model(tensors)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            n          += len(labels)

    return total_loss / max(n, 1), correct / max(n, 1)


def evaluate_per_class(model, loader, device, num_classes):
    model.eval()
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    with torch.no_grad():
        for tensors, labels in loader:
            tensors = tensors.to(device)
            labels  = labels.to(device)
            preds   = model(tensors).argmax(dim=1)
            for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                if p == l:
                    tp[l] += 1
                else:
                    fp[p] += 1
                    fn[l] += 1

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        prec = tp[i] / max(tp[i] + fp[i], 1)
        rec  = tp[i] / max(tp[i] + fn[i], 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-6)
        results[name] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "support":   tp[i] + fn[i],
        }
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PRIME CNN classifier")
    parser.add_argument("--data",       default="data/crops/labeled_crops")
    parser.add_argument("--config",     default="config/config.yaml")
    parser.add_argument("--output",     default="models/cnn/prime_classifier.pth")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch",      type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--input-size", type=int,   default=None,
                        help="128 (default) or 224. Try both, pick winner on val.")
    parser.add_argument("--val-frac",   type=float, default=0.20)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg    = load_config(args.config)
    logger = get_logger("train_cnn",
                        cfg.get("logging", "log_path", default="logs/prime.log"),
                        cfg.get("logging", "level",    default="INFO"))

    device      = args.device  or cfg.device
    epochs      = args.epochs  or cfg.get("cnn", "epochs",        default=30)
    batch_size  = args.batch   or cfg.get("cnn", "batch_size",    default=32)
    lr          = args.lr      or cfg.get("cnn", "learning_rate", default=5e-4)
    patience    = cfg.get("cnn", "early_stopping_patience", default=5)
    input_size  = args.input_size or cfg.get("cnn", "input_size", default=128)
    num_classes = cfg.get("cnn", "num_classes", default=5)

    print("=" * 55)
    print("PRIME — CNN Training")
    print("=" * 55)
    print(f"  Data       : {args.data}")
    print(f"  Output     : {args.output}")
    print(f"  Device     : {device}")
    print(f"  Input size : {input_size}x{input_size}")
    print(f"  Epochs     : {epochs}  Batch: {batch_size}  LR: {lr}")
    print(f"  Val frac   : {int(args.val_frac*100)}%")
    print()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    print("Scanning labeled crops...")
    all_samples = load_samples(data_dir)
    if not all_samples:
        print("ERROR: No labeled crops found. Run label_crops_standalone.py first.")
        sys.exit(1)
    print(f"  Total: {len(all_samples)} crops\n")

    train_samples, val_samples = train_val_split(all_samples, args.val_frac, args.seed)
    print(f"  Train: {len(train_samples)}  Val: {len(val_samples)}\n")

    train_ds = CropDataset(train_samples, input_size, augment=True)
    val_ds   = CropDataset(val_samples,   input_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print("Computing class weights...")
    class_weights = compute_class_weights(train_samples, num_classes, device)
    print()

    print("Building model...")
    model = build_model(num_classes, device)
    print()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss  = float("inf")
    patience_count = 0
    history        = []

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
    print("-" * 62)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.1%}  "
              f"{val_loss:>8.4f}  {val_acc:>6.1%}  {current_lr:>8.2e}  "
              f"({elapsed:.1f}s)")

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
        })

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(), output_path)
            print(f"         → saved best (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\nEarly stopping — no improvement for {patience} epochs.")
                break

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Model saved   : {output_path}")

    # Per-class breakdown on validation set
    print("\nLoading best model for per-class evaluation...")
    model.load_state_dict(torch.load(output_path, map_location=device))
    per_class = evaluate_per_class(model, val_loader, device, num_classes)

    print(f"\n{'Class':<22}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'Support':>7}")
    print("-" * 58)
    for name, m in per_class.items():
        print(f"{name:<22}  {m['precision']:>9.4f}  {m['recall']:>6.4f}  "
              f"{m['f1']:>6.4f}  {m['support']:>7}")

    fod_recall = per_class.get("fod", {}).get("recall", 0.0)
    if fod_recall < 0.85:
        print(f"\nWARNING: FOD recall = {fod_recall:.4f} — below 0.85 target.")
        print("  Options: collect more FOD crops, lower conf threshold, more augmentation.")

    # Save summary
    summary = {
        "best_val_loss":  round(best_val_loss, 4),
        "epochs_trained": len(history),
        "input_size":     input_size,
        "per_class":      per_class,
        "history":        history,
    }
    summary_path = output_path.parent / "train_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining summary → {summary_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
