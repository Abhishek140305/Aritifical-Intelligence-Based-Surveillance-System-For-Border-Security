"""Optional face-recognition and license-plate OCR modules.

Models are deliberately lazy-loaded: edge capture should continue when a module is
not deployed, while the console makes the missing capability visible.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class FaceRecognitionService:
    def __init__(self, watchlist_path: str) -> None:
        self.names: list[str] = []
        self.embeddings = np.empty((0, 512), dtype=np.float32)
        self.app = None
        path = Path(watchlist_path)
        if path.exists():
            data = np.load(path, allow_pickle=False)
            self.names, self.embeddings = data["names"].tolist(), data["embeddings"].astype(np.float32)
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(320, 320))
        except Exception as exc:
            print(f"[frs] disabled: {exc}")

    def match(self, person_crop: np.ndarray, threshold: float = 0.48) -> Optional[tuple[str, float]]:
        if self.app is None or not len(self.embeddings) or person_crop.size == 0:
            return None
        faces = self.app.get(person_crop)
        if not faces:
            return None
        embedding = faces[0].normed_embedding
        db = self.embeddings / np.maximum(np.linalg.norm(self.embeddings, axis=1, keepdims=True), 1e-9)
        scores = db @ embedding
        index = int(np.argmax(scores))
        return (self.names[index], float(scores[index])) if scores[index] >= threshold else None


class ANPRService:
    PLATE_PATTERN = re.compile(r"^[A-Z0-9]{5,12}$")
    def __init__(self) -> None:
        self.reader = None
        try:
            import easyocr
            self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception as exc:
            print(f"[anpr] disabled: {exc}")

    def read_plate(self, vehicle_crop: np.ndarray) -> Optional[tuple[str, float]]:
        if self.reader is None or vehicle_crop.size == 0:
            return None
        # A detector may replace this broad ROI in a specialized ANPR deployment.
        h = vehicle_crop.shape[0]
        roi = vehicle_crop[int(h * 0.45):]
        results = self.reader.readtext(roi, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        for _, text, confidence in results:
            normalized = re.sub(r"[^A-Z0-9]", "", text.upper())
            if self.PLATE_PATTERN.fullmatch(normalized):
                return normalized, float(confidence)
        return None
