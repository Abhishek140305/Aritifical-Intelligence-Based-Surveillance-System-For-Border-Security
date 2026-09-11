"""Night-time image enhancement routines."""
from __future__ import annotations

import cv2
import numpy as np


class LowLightPreprocessor:
    def __init__(self, enabled: bool = False, gamma: float = 1.25) -> None:
        self.enabled = enabled
        self.gamma = max(gamma, 0.1)
        self.clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        # LUT avoids a per-pixel power calculation on each video frame.
        inverse_gamma = 1.0 / self.gamma
        self.gamma_lut = np.array([((i / 255.0) ** inverse_gamma) * 255 for i in range(256)], dtype=np.uint8)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or frame is None:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return cv2.LUT(enhanced, self.gamma_lut)
