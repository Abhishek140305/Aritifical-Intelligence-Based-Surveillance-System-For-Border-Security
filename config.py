"""Configuration for an Intelligent Border Video Analytics Platform edge node.

All operational values can be overridden with environment variables.  Keep secrets
and camera URLs out of source control by placing them in a local `.env` file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple

Point = Tuple[int, int]


def _polygon(value: str) -> List[Point]:
    """Read BORDER_POLYGON as JSON, falling back to a conservative demo zone."""
    try:
        points = json.loads(value)
        return [(int(x), int(y)) for x, y in points]
    except (ValueError, TypeError):
        return [(220, 120), (1060, 120), (1180, 700), (80, 700)]


@dataclass(slots=True)
class Settings:
    camera_id: str = os.getenv("CAMERA_ID", "border-cam-01")
    rtsp_url: str = os.getenv("RTSP_URL", "")
    fallback_video: str = os.getenv("FALLBACK_VIDEO", "")
    model_path: str = os.getenv("YOLO_MODEL", "yolo11n.pt")
    tracker: str = os.getenv("YOLO_TRACKER", "bytetrack.yaml")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
    frame_buffer_size: int = int(os.getenv("FRAME_BUFFER_SIZE", "3"))
    process_fps: float = float(os.getenv("PROCESS_FPS", "8"))
    enable_low_light: bool = os.getenv("ENABLE_LOW_LIGHT", "false").lower() == "true"
    gamma: float = float(os.getenv("GAMMA", "1.25"))
    border_polygon: List[Point] = field(default_factory=lambda: _polygon(os.getenv("BORDER_POLYGON", "")))
    min_breach_speed_px_s: float = float(os.getenv("MIN_BREACH_SPEED_PX_S", "8"))
    watchlist_path: str = os.getenv("WATCHLIST_PATH", "watchlist.npz")
    anpr_watchlist: set[str] = field(default_factory=lambda: {
        value.strip().upper() for value in os.getenv("ANPR_WATCHLIST", "").split(",") if value.strip()
    })
    alert_webhook: str = os.getenv("ALERT_WEBHOOK", "http://127.0.0.1:8000/api/alerts")
    # 6 KB JPEG becomes ~8 KB after Base64, leaving room for alert metadata under 10 KB.
    snapshot_max_bytes: int = int(os.getenv("SNAPSHOT_MAX_BYTES", "6000"))
    cooldown_seconds: float = float(os.getenv("ALERT_COOLDOWN_SECONDS", "12"))


settings = Settings()
