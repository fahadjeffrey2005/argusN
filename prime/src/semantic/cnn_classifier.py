"""
PRIME CNN Classifier
MobileNetV3-Small modified to accept 4-channel input.
5-class output: fod, shadow, runway_marking, strobe_light, clean_tarmac.

Key modification:
  The standard MobileNetV3-Small first conv expects 3 channels.
  We extend it to 4 channels by copying the pretrained 3-channel weights
  and initialising the 4th channel (flow magnitude) near zero.
  The model learns the contribution of the flow channel during fine-tuning.

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

        self.device = cfg.device
        self.model_path = cfg.get("cnn", "model_path", default="models/cnn/prime_classifier.pth")
        self.pretrained_path = cfg.get("cnn", "pretrained_path", default="models/cnn/mobilenetv3_small.pth")
        self.input_size = cfg.get("cnn", "input_size", default=128)
        self.num_classes = cfg.get("cnn", "num_classes", default=5)
        self.conf_threshold = cfg.get("cnn", "confidence_threshold", default=0.6)
        self.both_tag_bonus = cfg.get("cnn", "both_tag_bonus", default=0.1)

        self.model = None
        self._load_model()

    def _build_model(self) -> nn.Module:
        """
        Build MobileNetV3-Small with 4-channel input and 5-class head.
        """
        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import mobilenet_v3_small
            model = mobilenet_v3_small(pretrained=True)

        # ── Modify first conv: 3 → 4 channels ──────────────────────────
        original_conv = model.features[0][0]  # Conv2d(3, 16, ...)

        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        # Copy pretrained 3-channel weights
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original_conv.weight.data
            # Initialise 4th channel (flow) near zero — model learns it
            nn.init.constant_(new_conv.weight[:, 3:, :, :], 0.0)
            if original_conv.bias is not None:
                new_conv.bias = original_conv.bias

        model.features[0][0] = new_conv

        # ── Replace classification head: ImageNet-1000 → 5 classes ──────
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, self.num_classes)

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

    def classify(
        self,
        crop_4ch: np.ndarray,
        tag: str = "yolo_only"
    ) -> dict:
        """
        Classify one 4-channel crop.

        Args:
            crop_4ch: (4, 128, 128) float32 [0-1]
            tag:      source tag from PrimeFusion ("both", "yolo_only", "flow_only")

        Returns:
            result dict:
                class_id:    int (0-4)
                class_name:  str
                confidence:  float
                is_fod:      bool
                tag:         str (passed through)
        """
        if self.model is None:
            return {"class_id": -1, "class_name": "unknown", "confidence": 0.0, "is_fod": False, "tag": tag}

        tensor = torch.from_numpy(crop_4ch).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        # Apply bonus for "both" tagged candidates (visual + physics agreement)
        effective_conf = confidence
        if tag == "both" and class_id == FOD_CLASS_ID:
            effective_conf = min(1.0, confidence + self.both_tag_bonus)

        is_fod = (class_id == FOD_CLASS_ID) and (effective_conf >= self.conf_threshold)

        return {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown",
            "confidence": round(effective_conf, 4),
            "raw_confidence": round(confidence, 4),
            "all_probs": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))},
            "is_fod": is_fod,
            "tag": tag
        }

    def classify_batch(
        self,
        crops_and_candidates: list
    ) -> list:
        """
        Classify a batch of (crop_4ch, candidate) pairs.

        Returns list of (classification_result, candidate) pairs.
        Skips entries where crop_4ch is None (degenerate regions).
        """
        results = []
        for crop_4ch, candidate in crops_and_candidates:
            if crop_4ch is None:
                continue
            tag = candidate.get("tag", "yolo_only")
            classification = self.classify(crop_4ch, tag)
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
            f"conf_threshold={self.conf_threshold}, "
            f"device={self.device})"
        )
