# Intelligent Border Video Analytics Platform (IBVAP) 

IBVAP is a modular Python MVP that converts one RTSP camera (or a local fallback video) into an edge analytics node. It is intended for authorized, lawful perimeter security operations. Configure retention, access controls, human review, audit trails, and local privacy/legal requirements before deployment.

## What is included

- A threaded **latest-frame RTSP ingestion** buffer with reconnect handling.
- Toggleable LAB/CLAHE + gamma low-light enhancement.
- YOLO detection plus persistent ByteTrack/BoT-SORT IDs, polygon entry detection, direction and pixel-speed estimation.
- Optional InsightFace cosine-similarity watchlist matching and EasyOCR vehicle plate reading.
- Conservative crawling/climbing heuristic; use a calibrated YOLO-Pose model for operational behavior classification.
- JSON alerts with compact evidence thumbnails, dispatched without blocking inference.
- FastAPI REST/WebSocket C2 backend and a responsive browser dashboard with annotated previews.

## Install

Python 3.10+ is recommended. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional modules for face matching / ANPR:
pip install insightface onnxruntime easyocr
```

For NVIDIA deployments, install a CUDA-compatible PyTorch build before `ultralytics`. The default `yolo11n.pt` will be fetched by Ultralytics on its first use; place approved weights locally and set `YOLO_MODEL` for offline deployments.

## Configure and run

Set a source. A local `FALLBACK_VIDEO` makes a safe initial test:

```bash
export CAMERA_ID=border-north-01
export RTSP_URL='rtsp://user:password@camera.example:554/stream1'
# Or: export FALLBACK_VIDEO='/absolute/path/to/test.mp4'
export BORDER_POLYGON='[[220,120],[1060,120],[1180,700],[80,700]]'
export ENABLE_LOW_LIGHT=true
```

Start C2 first, then the edge process in another terminal:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
python main.py
```

Open `http://127.0.0.1:8000`. The C2 API exposes `GET /health`, `GET /api/alerts`, `POST /api/alerts`, `POST /api/frame`, and `WS /ws`.

## Watchlists

FRS expects a local NumPy archive such as `watchlist.npz` with `names` (string array) and `embeddings` (N×512 float32, generated from consented reference imagery using the same ArcFace model). Do not place biometric images or embeddings in source control. Set `ANPR_WATCHLIST=KA01AB1234,DL01AA0001` to produce ANPR alerts only for listed normalized registrations.

## Production notes

- `BORDER_POLYGON` uses image pixels, ordered around the restricted zone. Calibrate it per camera. Pixel speed is not a physical speed; apply camera homography/calibration before using it as a distance measurement.
- The alert thumbnail defaults to a 6 KB JPEG before Base64 encoding, so a normal alert remains below the requested 10 KB envelope. The API keeps only 200 alerts in memory; use an authenticated, durable broker/database for production.
- Keep RTSP credentials in a secret manager, put C2 behind TLS and authentication, restrict network ingress, and restrict/watch audit access to face/plate data.
- The `CRAWLING_DETECTED` geometry classifier is deliberately only a triage signal. Validate all alerts with a human operator and replace/tune it using representative local footage before operational use.
