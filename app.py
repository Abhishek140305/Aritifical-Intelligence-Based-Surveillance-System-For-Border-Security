"""FastAPI Command-and-Control backend and zero-build live dashboard."""
from __future__ import annotations

import base64
from collections import deque
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="IBVAP C2", version="0.1.0")
alerts: deque[dict[str, Any]] = deque(maxlen=200)
clients: set[WebSocket] = set()
latest_frame: bytes | None = None


class Alert(BaseModel):
    model_config = ConfigDict(extra="allow")
    timestamp: str
    camera_id: str
    event_type: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[int] = Field(min_length=4, max_length=4)
    snapshot_thumbnail_b64: str = ""


class Frame(BaseModel):
    image_b64: str = Field(min_length=10, max_length=3_000_000)


async def broadcast(payload: dict[str, Any]) -> None:
    dead: list[WebSocket] = []
    for client in clients:
        try:
            await client.send_json(payload)
        except Exception:
            dead.append(client)
    for client in dead:
        clients.discard(client)


@app.post("/api/alerts", status_code=202)
async def receive_alert(alert: Alert) -> dict[str, bool]:
    payload = alert.model_dump()
    # Permit useful structured extensions (track ID, identity, vector) from the edge node.
    alerts.appendleft(payload)
    await broadcast({"type": "alert", "data": payload})
    return {"accepted": True}


@app.get("/api/alerts")
async def list_alerts(limit: int = 50) -> list[dict[str, Any]]:
    if not 1 <= limit <= 200:
        raise HTTPException(422, "limit must be between 1 and 200")
    return list(alerts)[:limit]


@app.post("/api/frame", status_code=202)
async def receive_frame(frame: Frame) -> dict[str, bool]:
    global latest_frame
    try:
        raw = base64.b64decode(frame.image_b64, validate=True)
    except ValueError as exc:
        raise HTTPException(422, "invalid JPEG Base64") from exc
    if not raw.startswith(b"\xff\xd8"):
        raise HTTPException(422, "preview must be a JPEG")
    latest_frame = raw
    return {"accepted": True}


@app.get("/api/live-frame")
async def live_frame() -> Response:
    if latest_frame is None:
        raise HTTPException(404, "waiting for edge preview")
    return Response(latest_frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept(); clients.add(websocket)
    try:
        await websocket.send_json({"type": "history", "data": list(alerts)})
        while True:
            await websocket.receive_text()  # allows simple client heartbeat messages
    except WebSocketDisconnect:
        clients.discard(websocket)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "connected_dashboards": len(clients), "alert_buffer": len(alerts)}


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    return DASHBOARD_HTML


DASHBOARD_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IBVAP Command & Control</title><style>
body{margin:0;background:#081019;color:#e8f0fa;font:14px system-ui,sans-serif}.bar{padding:16px 24px;background:#0d1b29;border-bottom:1px solid #203348}.grid{display:grid;grid-template-columns:minmax(300px,2fr) minmax(300px,1fr);gap:16px;padding:16px}.card{background:#0d1b29;border:1px solid #203348;border-radius:8px;padding:14px}.feed{height:440px;display:grid;place-items:center;background:#05090e;color:#8294aa}.incident{border-left:4px solid #e65050;margin:9px 0;padding:9px;background:#132438}.badge{display:inline-block;padding:3px 7px;border-radius:12px;background:#d63636;font-weight:700;font-size:11px}.meta{color:#a9b9cc;margin:4px 0}.thumb{width:110px;max-height:70px;object-fit:cover;float:right;margin-left:9px}#log{max-height:520px;overflow:auto}@media(max-width:750px){.grid{grid-template-columns:1fr}.feed{height:240px}}</style></head>
<body><div class="bar"><b>IBVAP</b> &nbsp; Intelligent Border Video Analytics <span id="status" class="meta">● connecting</span></div><main class="grid"><section class="card"><h3>Live annotated feed</h3><div class="feed"><img id="feed" alt="Waiting for edge video" style="max-width:100%;max-height:100%;display:none"></div></section><aside class="card"><h3>Live incident log</h3><div id="log"></div></aside></main>
<script>const log=document.querySelector('#log'),status=document.querySelector('#status'),feed=document.querySelector('#feed');function add(a){let i=document.createElement('article');i.className='incident';let img=a.snapshot_thumbnail_b64?`<img class="thumb" src="data:image/jpeg;base64,${a.snapshot_thumbnail_b64}">`:'';i.innerHTML=`${img}<span class="badge">${a.event_type}</span><div class="meta">${a.camera_id} · confidence ${(a.confidence*100).toFixed(1)}%</div><div class="meta">${new Date(a.timestamp).toLocaleString()} · bbox [${a.bbox}]</div>`;log.prepend(i)}function refresh(){feed.src='/api/live-frame?'+Date.now();feed.style.display='block'}setInterval(refresh,700);let p=location.protocol==='https:'?'wss':'ws';let ws=new WebSocket(`${p}://${location.host}/ws`);ws.onopen=()=>{status.textContent='● live';status.style.color='#5ce08a'};ws.onmessage=e=>{let m=JSON.parse(e.data);if(m.type==='history')m.data.reverse().forEach(add);else add(m.data)};ws.onclose=()=>{status.textContent='● reconnecting';setTimeout(()=>location.reload(),2000)};</script></body></html>'''
