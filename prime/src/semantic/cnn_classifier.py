"""
PRIME CNN Classifier
MobileNetV3-Small modified to accept 4-channel input.
5-class output: fod, shadow, runway_marking, strobe_light, clean_tarmac.

Key modification:
  The standard MobileNetV3-Small first conv expects 3 channels.
  We extend it to 4 channels by copying the pretrained 3-channel weights
  and initialising the 4th channel (frame-diff) near zero.
  The first conv and classifier head are trainable; the backbone is frozen.

Threshold design:
  confidence_threshold  (default 0.25): hard-reject floor. Only drop a
      candidate if the CNN is fairly confident it is NOT FOD. Everything
      ambiguous survives to the temporal tracker.
  fast_path_threshold   (default 0.90): if CNN confidence for FOD exceeds
      this, skip the 2-frame prefilter and enter the 3-frame tracker
      directly. High-certainty detections get immediate classification.

Only class 0 (fod) raises an alert — all others are discarded.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from src.utils.config_loader import Config
from src.utils.logger import get_logger

CLASS_NAMES = ["fod", "shadow", "runway_marking", "strobe_light", "clean_tarmac"]
FOD_CLASS_ID = 0


class CNNClassifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = get_logger(
            "cnn_classifier",
            cfg.get("logging", "log_path", default="logs/prime.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.device      = cfg.device
        self.model_path  = cfg.get("cnn", "model_path",  default="models/cnn/prime_classifier.pth")
        self.input_size  = cfg.get("cnn", "input_size",  default=128)
        self.num_classes = cfg.get("cnn", "num_classes", default=5)

        # Hard-reject floor — only kill candidates the CNN is confident are NOT FOD
        self.conf_threshold      = cfg.get("cnn", "confidence_threshold", default=0.25)
        # Fast-path ceiling — skip 2-frame prefilter when CNN is near-certain about FOD
        self.fast_path_threshold = cfg.get("cnn", "fast_path_threshold",  default=0.90)

        self.model = None
        self._load_model()

    def _build_model(self) -> nn.Module:
        """
        Build MobileNetV3-Small with 4-channel input and 5-class head.
        Backbone is frozen except the first conv layer and classifier head.
        """
        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import mobilenet_v3_small
            model = mobilenet_v3_small(pretrained=True)

        # ── Freeze entire backbone first ─────────────────────────────────
        for param in model.parameters():
            param.requires_grad = False

        # ── Modify first conv: 3 → 4 channels (must unfreeze to learn 4th ch) ──
        original_conv = model.features[0][0]  # Conv2d(3, 16, ...)
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        # Copy pretrained RGB weights; initialise frame-diff channel near zero
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original_conv.weight.data
            nn.init.constant_(new_conv.weight[:, 3:, :, :], 0.0)
            if original_conv.bias is not None:
                new_conv.bias = nn.Parameter(original_conv.bias.data.clone())
        model.features[0][0] = new_conv

        # Unfreeze first conv — must learn what frame-diff means
        for param in model.features[0][0].parameters():
            param.requires_grad = True

        # ── Replace classification head: ImageNet-1000 → 5 classes ──────
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, self.num_classes)

        # Unfreeze classifier head
        for param in model.classifier.parameters():
            param.requires_grad = True

        return model

    def _load_model(self):
        """
        Load trained weights if they exist.
        Otherwise build the architecture ready for training.
        """
        self.model = self._build_model()

        if Path(self.model_path).exists():
            try:
                state = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.logger.info(f"PRIME CNN loaded from {self.model_path}")
            except Exception as e:
                self.logger.error(f"CNN load failed: {e}")
        else:
            self.logger.warning(
                f"No trained weights at {self.model_path} — "
                f"model architecture ready, run train_cnn.py first"
            )

        self.model = self.model.to(self.device)
        self.model.eval()

    def classify(self, crop_4ch: np.ndarray) -> dict:
        """
        Classify one 4-channel crop.

        Args:
            crop_4ch: (4, H, W) float32 [0-1]

        Returns dict with:
            class_id:    int (0-4)
            class_name:  str
            confidence:  float (FOD class probability)
            is_fod:      bool  (True if class==FOD and conf >= conf_threshold)
            fast_path:   bool  (True if conf >= fast_path_threshold — skip prefilter)
            all_probs:   dict  {class_name: prob}
        """
        tensor = torch.from_numpy(crop_4ch).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        class_id   = int(np.argmax(probs))
        confidence = float(probs[class_id])

        # FOD-specific confidence (even if not the argmax class)
        fod_conf = float(probs[FOD_CLASS_ID])

        is_fod    = (class_id == FOD_CLASS_ID) and (confidence >= self.conf_threshold)
        fast_path = is_fod and (confidence >= self.fast_path_threshold)

        return {
            "class_id":   class_id,
            "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown",
            "confidence": round(confidence, 4),
            "fod_conf":   round(fod_conf, 4),
            "is_fod":     is_fod,
            "fast_path":  fast_path,
            "all_probs":  {CLASS_NAMES[i]: round(float(probs[i]), 4)
                           for i in range(len(CLASS_NAMES))},
        }

    def classify_batch(self, crops_and_candidates: list) -> list:
        """
        Classify a batch of (crop_4ch, candidate) pairs.

        Returns list of (classification_result, candidate) pairs.
        Skips entries where crop_4ch is None (degenerate regions).
        """
        results = []
        for crop_4ch, candidate in crops_and_candidates:
            if crop_4ch is None:
                continue
            classification = self.classify(crop_4ch)
            results.append((classification, candidate))
        return results

    def save_weights(self, path: str = None):
        """Save model weights."""
        path = path or self.model_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"CNN weights saved to {path}")

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def __repr__(self):
        return (
            f"CNNClassifier("
            f"classes={self.num_classes}, "
            f"reject_below={self.conf_threshold}, "
            f"fast_path_above={self.fast_path_threshold}, "
            f"device={self.device})"
        )
