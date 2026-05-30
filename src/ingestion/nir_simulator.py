"""
ARGUS-N NIR Simulator
Simulates a Near-Infrared (850nm) camera output from an RGB frame.
Used during development when only RGB cameras are available.
On real hardware this is replaced by the actual NIR camera stream.

Physics basis:
- Asphalt/tarmac absorbs NIR strongly → appears dark
- Metals (washers, bolts) reflect NIR strongly → appear bright
- Rubber/PVC wire insulation has different NIR reflectance than tarmac
- The red channel of RGB is the closest to NIR in a standard sensor

Simulation approach:
- Weight R channel heavily (closest to NIR band)
- Suppress blue channel (farthest from NIR)
- Apply CLAHE to enhance contrast between materials
- Add synthetic metallic specular response for flat objects
"""

import cv2
import numpy as np


class NIRSimulator:
    def __init__(self, clahe_clip: float = 3.0, clahe_grid: int = 8):
        """
        clahe_clip:  CLAHE clip limit — higher = more aggressive contrast
        clahe_grid:  CLAHE tile grid size
        """
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid)
        )

    def simulate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Convert BGR frame to simulated NIR grayscale image.

        Returns:
            nir: (H, W) uint8 — simulated NIR image
        """
        b = frame_bgr[:, :, 0].astype(np.float32)
        g = frame_bgr[:, :, 1].astype(np.float32)
        r = frame_bgr[:, :, 2].astype(np.float32)

        # NIR approximation: heavy red, some green, suppress blue
        # Weights tuned for tarmac/metal differentiation
        nir = (0.65 * r + 0.30 * g - 0.05 * b)
        nir = np.clip(nir, 0, 255).astype(np.uint8)

        # Apply CLAHE — enhances material contrast
        nir = self.clahe.apply(nir)

        return nir

    def to_bgr(self, nir: np.ndarray) -> np.ndarray:
        """Convert NIR grayscale to BGR for display/debugging."""
        return cv2.cvtColor(nir, cv2.COLOR_GRAY2BGR)

    def side_by_side(
        self,
        frame_bgr: np.ndarray,
        nir: np.ndarray
    ) -> np.ndarray:
        """
        Stack RGB and NIR side by side for visual comparison.
        Useful during development to verify NIR simulation quality.
        """
        nir_bgr = self.to_bgr(nir)
        h = frame_bgr.shape[0]
        nir_resized = cv2.resize(nir_bgr, (frame_bgr.shape[1], h))

        # Label each side
        cv2.putText(frame_bgr, "RGB", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(nir_resized, "NIR (simulated)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return np.hstack([frame_bgr, nir_resized])
