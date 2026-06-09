"""
ARGUS-N Config Loader
Loads and validates master config.
Exposes typed access throughout the pipeline.
"""

import yaml
from pathlib import Path


class Config:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

    def get(self, *keys, default=None):
        """
        Traverse nested keys safely.
        Usage: cfg.get('yolo', 'confidence_threshold')
        """
        val = self._cfg
        for key in keys:
            if not isinstance(val, dict) or key not in val:
                return default
            val = val[key]
        return val

    # ── Shortcuts ──────────────────────────────────────────

    @property
    def device(self) -> str:
        return self.get("device", default="cpu")

    @property
    def camera(self) -> dict:
        return self.get("camera", default={})

    @property
    def yolo(self) -> dict:
        return self.get("yolo", default={})

    @property
    def pipeline(self) -> dict:
        return self.get("pipeline", default={})

    @property
    def outputs(self) -> dict:
        return self.get("outputs", default={})

    def __repr__(self):
        return f"Config(device={self.device})"


def load_config(path: str = "config/config.yaml") -> Config:
    return Config(path)


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
    print(f"Device            : {cfg.device}")
    print(f"FPS               : {cfg.get('camera', 'fps')}")
    print(f"YOLO confidence   : {cfg.get('yolo', 'confidence_threshold')}")
    print(f"Freeze layers     : {cfg.get('yolo', 'freeze_layers')}")
