"""Edge-node capture and inference process. Start the API separately with app.py."""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from alert_engine import AlertDispatcher, FramePublisher
from config import Settings, settings
from detector import PERSON_CLASS, VEHICLE_CLASSES, Detection, Detector
from frs_anpr import ANPRService, FaceRecognitionService
from preprocessor import LowLightPreprocessor


class FrameGrabber:
    """A latest-frame queue isolates RTSP jitter from inference latency."""
    def __init__(self, source: str, buffer_size: int) -> None:
        self.source, self.frames = source, queue.Queue(maxsize=buffer_size)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="rtsp-grabber")

    def start(self) -> None: self.thread.start()
    def stop(self) -> None: self.running = False

    def _run(self) -> None:
        while self.running:
            # OpenCV requires device indices as integers; retain RTSP URLs and paths as strings.
            source = int(self.source) if self.source.strip().isdigit() else self.source
            capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not capture.isOpened():
                print("[ingest] stream unavailable; retrying in 3 seconds")
                time.sleep(3); continue
            print("[ingest] stream connected")
            while self.running:
                ok, frame = capture.read()
                if not ok:
                    print("[ingest] dropped stream; reconnecting")
                    break
                if self.frames.full():
                    try: self.frames.get_nowait()
                    except queue.Empty: pass
                try: self.frames.put_nowait(frame)
                except queue.Full: pass
            capture.release()
            time.sleep(1)

    def latest(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try: return self.frames.get(timeout=timeout)
        except queue.Empty: return None


def crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    return frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]


def run(cfg: Settings = settings) -> None:
    source = cfg.rtsp_url or cfg.fallback_video
    if not source:
        raise ValueError("Set RTSP_URL or FALLBACK_VIDEO before starting the edge node.")
    grabber = FrameGrabber(source, cfg.frame_buffer_size); grabber.start()
    preprocessor, detector = LowLightPreprocessor(cfg.enable_low_light, cfg.gamma), Detector(cfg)
    frs, anpr = FaceRecognitionService(cfg.watchlist_path), ANPRService()
    alerts = AlertDispatcher(cfg.alert_webhook, cfg.cooldown_seconds, cfg.snapshot_max_bytes)
    preview = FramePublisher(cfg.alert_webhook)
    interval, last_processed = 1 / max(cfg.process_fps, 0.1), 0.0
    try:
        while True:
            frame = grabber.latest()
            if frame is None or time.monotonic() - last_processed < interval: continue
            last_processed = time.monotonic()
            enhanced = preprocessor.apply(frame)
            detections = detector.detect(enhanced)
            for det in detections:
                subject = crop(enhanced, det.bbox)
                if det.entered_zone:
                    alerts.emit(cfg.camera_id, "TRIPWIRE_BREACH", det.confidence, det.bbox, enhanced,
                                track_id=det.track_id, speed_px_s=round(det.speed_px_s, 2), vector=det.vector)
                if det.class_id == PERSON_CLASS and det.behavior == "CRAWLING_DETECTED":
                    alerts.emit(cfg.camera_id, "CRAWLING_DETECTED", det.confidence, det.bbox, enhanced, track_id=det.track_id)
                if det.class_id == PERSON_CLASS:
                    match = frs.match(subject)
                    if match:
                        alerts.emit(cfg.camera_id, "FRS_MATCH", match[1], det.bbox, enhanced, identity=match[0], track_id=det.track_id)
                elif det.class_id in VEHICLE_CLASSES:
                    plate = anpr.read_plate(subject)
                    if plate and plate[0] in cfg.anpr_watchlist:
                        alerts.emit(cfg.camera_id, "ANPR_ALERT", plate[1], det.bbox, enhanced, plate=plate[0], track_id=det.track_id)
            preview.publish(detector.annotate(enhanced, detections))
    except KeyboardInterrupt:
        print("[edge] stopping")
    finally:
        grabber.stop()


if __name__ == "__main__":
    run()
