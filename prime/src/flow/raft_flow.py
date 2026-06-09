"""
ARGUS-N RAFT Optical Flow
Computes optical flow between frame T and frame T-1.
Outputs a flow map showing direction and speed of every pixel.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from src.utils.config_loader import Config
from src.utils.logger import get_logger


class RAFTFlow:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device
        self.logger = get_logger(
            "raft_flow",
            cfg.get("logging", "log_path", default="logs/argus.log"),
            cfg.get("logging", "level", default="INFO")
        )

        self.model_path = cfg.get("raft", "model_path", default="models/raft/raft_small.pth")
        self.iterations = cfg.get("raft", "iterations", default=12)
        self.input_width = cfg.get("raft", "input_size", "width", default=1920)
        self.input_height = cfg.get("raft", "input_size", "height", default=1080)

        self.model = None
        self.prev_frame_tensor = None

        self._load_model()

    def _load_model(self):
        """
        Load RAFT-Small model.
        Falls back to Farneback optical flow if RAFT weights not found.
        Farneback is CPU based — useful for development without model weights.
        """
        if Path(self.model_path).exists():
            try:
                # RAFT model loading
                # Requires: pip install torch torchvision
                # Weights from: https://github.com/princeton-vl/RAFT
                from torchvision.models.optical_flow import raft_small
                self.model = raft_small(pretrained=False)
                state = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state)
                self.model = self.model.to(self.device)
                self.model.eval()
                self.logger.info(f"RAFT-Small loaded from {self.model_path}")
                self.use_raft = True
            except Exception as e:
                self.logger.warning(f"RAFT load failed: {e} — falling back to Farneback")
                self.use_raft = False
        else:
            self.logger.warning(
                f"RAFT weights not found at {self.model_path} "
                f"— falling back to Farneback for development"
            )
            self.use_raft = False

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert BGR frame to normalised RGB tensor.
        Shape: (1, 3, H, W) float32 in range [0, 1]
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.input_width, self.input_height))
        tensor = torch.from_numpy(frame_resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _raft_flow(self, frame1_tensor: torch.Tensor, frame2_tensor: torch.Tensor) -> np.ndarray:
        """Run RAFT model and return flow as numpy array (H, W, 2)."""
        with torch.no_grad():
            flow_predictions = self.model(frame1_tensor, frame2_tensor, num_flow_updates=self.iterations)
            flow = flow_predictions[-1]  # final prediction
        flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return flow_np

    def _farneback_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Farneback dense optical flow.
        Development fallback — no weights needed.
        Returns flow as numpy array (H, W, 2)
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.resize(gray1, (self.input_width, self.input_height))
        gray2 = cv2.resize(gray2, (self.input_width, self.input_height))
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        return flow  # (H, W, 2)

    def compute(self, frame: np.ndarray):
        """
        Compute optical flow between previous frame and current frame.
        First call stores frame and returns None — no previous frame yet.
        Returns flow map (H, W, 2) — dx and dy per pixel.
        """
        if self.prev_frame_tensor is None:
            # First frame — store and return None
            if self.use_raft:
                self.prev_frame_tensor = self._preprocess(frame)
            else:
                self.prev_frame_tensor = frame.copy()
            self.logger.debug("First frame stored — flow computation starts next frame")
            return None

        if self.use_raft:
            curr_tensor = self._preprocess(frame)
            flow = self._raft_flow(self.prev_frame_tensor, curr_tensor)
            self.prev_frame_tensor = curr_tensor
        else:
            flow = self._farneback_flow(self.prev_frame_tensor, frame)
            self.prev_frame_tensor = frame.copy()

        return flow  # (H, W, 2)

    def reset(self):
        """Reset previous frame — call at start of each sweep."""
        self.prev_frame_tensor = None
        self.logger.info("RAFT flow reset — ready for new sweep")

    def __repr__(self):
        return (
            f"RAFTFlow("
            f"backend={'raft' if self.use_raft else 'farneback'}, "
            f"device={self.device}, "
            f"iterations={self.iterations})"
        )
