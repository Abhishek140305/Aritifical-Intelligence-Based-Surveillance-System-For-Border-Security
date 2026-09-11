"""Bounded alert serialization and reliable non-blocking HTTP dispatch."""
from __future__ import annotations

import base64
import json
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict

import cv2
import numpy as np
import requests


def thumbnail_base64(frame: np.ndarray, max_bytes: int) -> str:
    """JPEG-compress until the encoded image is safely within the alert budget."""
    # Cap caller configuration: Base64 has ~33% overhead and metadata needs headroom.
    max_bytes = min(max(max_bytes, 500), 6_000)
    thumbnail = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    for quality in (55, 40, 28, 18):
        ok, encoded = cv2.imencode(".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok and len(encoded) <= max_bytes:
            return base64.b64encode(encoded).decode("ascii")
    return ""  # preserve the <10KB contract even for noisy imagery


class AlertDispatcher:
    def __init__(self, webhook: str, cooldown: float, snapshot_max_bytes: int) -> None:
        self.webhook, self.cooldown, self.snapshot_max_bytes = webhook, cooldown, snapshot_max_bytes
        self.queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=100)
        self.last_sent: dict[str, float] = {}
        self.thread = threading.Thread(target=self._send_loop, daemon=True, name="alert-dispatcher")
        self.thread.start()

    def emit(self, camera_id: str, event_type: str, confidence: float, bbox: tuple[int, int, int, int], frame: np.ndarray, **extra: Any) -> None:
        key = f"{camera_id}:{event_type}:{extra.get('track_id', '')}"
        now = time.monotonic()
        if now - self.last_sent.get(key, 0) < self.cooldown:
            return
        self.last_sent[key] = now
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(), "camera_id": camera_id,
            "event_type": event_type, "confidence": round(float(confidence), 4), "bbox": list(bbox),
            "snapshot_thumbnail_b64": thumbnail_base64(frame, self.snapshot_max_bytes), **extra,
        }
        # The expected payload is <10 KB. If deployment-specific metadata grows,
        # evidence is omitted rather than violating the transport contract.
        if len(json.dumps(payload, separators=(",", ":")).encode("utf-8")) >= 10_000:
            payload["snapshot_thumbnail_b64"] = ""
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            print("[alerts] queue full; dropping oldest alert")
            try: self.queue.get_nowait()
            except queue.Empty: pass
            self.queue.put_nowait(payload)

    def _send_loop(self) -> None:
        while True:
            payload = self.queue.get()
            try:
                response = requests.post(self.webhook, json=payload, timeout=2.5)
                response.raise_for_status()
            except requests.RequestException as exc:
                print(f"[alerts] dispatch failed: {exc}")


class FramePublisher:
    """Publishes a low-bandwidth annotated preview independently of alert traffic."""
    def __init__(self, alert_webhook: str) -> None:
        self.endpoint = alert_webhook.rsplit("/", 1)[0] + "/frame"
        self.queue: queue.Queue[str] = queue.Queue(maxsize=1)
        threading.Thread(target=self._send_loop, daemon=True, name="frame-publisher").start()

    def publish(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        preview = cv2.resize(frame, (640, max(1, int(h * 640 / w))), interpolation=cv2.INTER_AREA) if w > 640 else frame
        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if not ok:
            return
        if self.queue.full():
            try: self.queue.get_nowait()
            except queue.Empty: pass
        try: self.queue.put_nowait(base64.b64encode(encoded).decode("ascii"))
        except queue.Full: pass

    def _send_loop(self) -> None:
        while True:
            image_b64 = self.queue.get()
            try:
                requests.post(self.endpoint, json={"image_b64": image_b64}, timeout=1.5).raise_for_status()
            except requests.RequestException:
                pass  # C2 may be restarted; a newer frame will replace this one.
