"""
inspect_model.py — Inspect best.pt to understand exactly what we have.

Run from argusN/ on Ubuntu:
    python inspect_model.py

Tells you:
  - Model architecture (YOLOv8n / v11s / etc.)
  - Number of parameters
  - Classes it was trained on
  - Input size
  - Training metadata (if saved)
  - Runs a quick inference on a blank image to confirm it works
"""

from pathlib import Path
import sys

WEIGHTS = Path("yolofinetune/models/yolo/finetuned/best.pt")

if not WEIGHTS.exists():
    print(f"ERROR: {WEIGHTS} not found. Run from argusN/ root.")
    sys.exit(1)

print("\n" + "="*60)
print("  best.pt Inspector")
print("="*60)

# ── 1. Raw checkpoint metadata ───────────────────────────────
import torch
ckpt = torch.load(str(WEIGHTS), map_location="cpu")

print("\n[1] Checkpoint keys:")
for k in ckpt.keys():
    v = ckpt[k]
    if isinstance(v, dict):
        print(f"    {k}: dict with {len(v)} keys")
    elif isinstance(v, (int, float, str, bool)):
        print(f"    {k}: {v}")
    else:
        print(f"    {k}: {type(v).__name__}")

# ── 2. Training args (if saved) ──────────────────────────────
if "train_args" in ckpt:
    print("\n[2] Training arguments:")
    args = ckpt["train_args"]
    for k, v in (args.__dict__ if hasattr(args, "__dict__") else args).items():
        print(f"    {k}: {v}")
elif "args" in ckpt:
    print("\n[2] Training arguments:")
    args = ckpt["args"]
    for k, v in (args.__dict__ if hasattr(args, "__dict__") else dict(args)).items():
        print(f"    {k}: {v}")
else:
    print("\n[2] No training args saved in checkpoint.")

# ── 3. Ultralytics model info ────────────────────────────────
print("\n[3] Loading with Ultralytics YOLO...")
try:
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))

    print(f"\n    Model type   : {type(model.model).__name__}")
    print(f"    Task         : {model.task}")
    print(f"    Input size   : {model.overrides.get('imgsz', 'unknown')}")

    # Parameter count
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable    = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"    Total params : {total_params:,}")
    print(f"    Trainable    : {trainable:,}")
    print(f"    Model size   : {WEIGHTS.stat().st_size / 1e6:.1f} MB")

    # Classes
    print(f"\n    Classes ({len(model.names)}):")
    for idx, name in model.names.items():
        print(f"      {idx}: {name}")

    # Model YAML (architecture)
    if hasattr(model.model, 'yaml') and model.model.yaml:
        print(f"\n    Architecture YAML:")
        for k, v in model.model.yaml.items():
            if k not in ('backbone', 'head'):
                print(f"      {k}: {v}")

except Exception as e:
    print(f"    ERROR: {e}")

# ── 4. Quick inference test ──────────────────────────────────
print("\n[4] Quick inference test (blank 640x640 image)...")
try:
    import numpy as np
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(dummy, verbose=False)
    dets = results[0].boxes
    n = len(dets) if dets is not None else 0
    print(f"    Inference OK — {n} detections on blank image (expected 0)")
except Exception as e:
    print(f"    ERROR during inference: {e}")

# ── 5. Check for NWD / SPD-Conv ──────────────────────────────
print("\n[5] Architecture check (NWD / SPD-Conv)...")
try:
    model_str = str(model.model)
    has_spd = "SPD" in model_str or "space_to_depth" in model_str.lower()
    print(f"    SPD-Conv present : {'YES' if has_spd else 'NO — standard downsampling'}")
    print(f"    NWD loss         : cannot detect from weights alone (training-time setting)")
    print(f"    NOTE: If NWD was used during training, recall for tiny objects would be higher.")
except Exception as e:
    print(f"    ERROR: {e}")

print("\n" + "="*60)
print("  Done. Share this output so we can plan the retraining.")
print("="*60 + "\n")
