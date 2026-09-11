"""Detection, tracking, virtual-zone and behavior analysis."""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Tuple

import cv2
import numpy as np

from config import Point, Settings

PERSON_CLASS = 0
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck in COCO


@dataclass(slots=True)
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    label: str
    track_id: int | None
    speed_px_s: float = 0.0
    vector: Tuple[float, float] = (0.0, 0.0)
    entered_zone: bool = False
    behavior: str = "NORMAL"


class TrajectoryStore:
    """Keeps short per-ID trajectories and detects outside-to-inside crossings."""
    def __init__(self, polygon: Iterable[Point], min_speed: float) -> None:
        self.polygon = np.asarray(list(polygon), dtype=np.int32)
        self.min_speed = min_speed
        self.history: Dict[int, Deque[Tuple[float, float, float, bool]]] = defaultdict(lambda: deque(maxlen=20))

    def update(self, track_id: int, bbox: Tuple[int, int, int, int]) -> tuple[float, tuple[float, float], bool]:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        inside = cv2.pointPolygonTest(self.polygon, (cx, cy), False) >= 0
        now = time.monotonic()
        trail = self.history[track_id]
        speed, vector, entered = 0.0, (0.0, 0.0), False
        if trail:
            px, py, previous_time, was_inside = trail[-1]
            elapsed = max(now - previous_time, 0.001)
            vector = ((cx - px) / elapsed, (cy - py) / elapsed)
            speed = math.hypot(*vector)
            entered = inside and not was_inside and speed >= self.min_speed
        trail.append((cx, cy, now, inside))
        return speed, vector, entered


class Detector:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.trajectories = TrajectoryStore(cfg.border_polygon, cfg.min_breach_speed_px_s)
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(cfg.model_path)
        except Exception as exc:  # App remains observable even without model dependencies.
            print(f"[detector] YOLO unavailable: {exc}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.model is None:
            return []
        try:
            result = self.model.track(
                frame, persist=True, tracker=self.cfg.tracker, conf=self.cfg.confidence_threshold,
                classes=[PERSON_CLASS, *VEHICLE_CLASSES], verbose=False,
            )[0]
        except Exception as exc:
            print(f"[detector] inference failed: {exc}")
            return []
        if result.boxes is None or len(result.boxes) == 0:
            return []
        ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [None] * len(result.boxes)
        boxes = result.boxes.xyxy.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        names = result.names
        output: List[Detection] = []
        for box, confidence, class_id, track_id in zip(boxes, confidences, classes, ids):
            detection = Detection(tuple(box), float(confidence), int(class_id), str(names[class_id]), track_id)
            if track_id is not None:
                detection.speed_px_s, detection.vector, detection.entered_zone = self.trajectories.update(track_id, detection.bbox)
            if class_id == PERSON_CLASS:
                detection.behavior = self._classify_behavior(detection.bbox)
            output.append(detection)
        return output

    @staticmethod
    def _classify_behavior(bbox: Tuple[int, int, int, int]) -> str:
        """Conservative geometry fallback; replace with YOLO-pose in production tuning."""
        x1, y1, x2, y2 = bbox
        width, height = max(x2 - x1, 1), max(y2 - y1, 1)
        ratio = width / height
        if ratio > 1.25:
            return "CRAWLING_DETECTED"
        if ratio < 0.28 and height > 100:
            return "CLIMBING_SUSPECTED"
        return "NORMAL"

    def annotate(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        canvas = frame.copy()
        cv2.polylines(canvas, [self.trajectories.polygon], True, (0, 0, 255), 2)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 0, 255) if det.entered_zone or det.behavior != "NORMAL" else (0, 200, 0)
            text = f"{det.label} #{det.track_id or '-'} {det.confidence:.2f}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, text, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
        return canvas
