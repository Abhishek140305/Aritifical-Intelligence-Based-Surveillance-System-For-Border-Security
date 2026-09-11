"""IBVAP: single-file edge analytics node and FastAPI C2 dashboard.

Run C2:   uvicorn ibvap:app --host 0.0.0.0 --port 8000
Run edge: python ibvap.py edge
"""
from __future__ import annotations

import base64
import json
import math
import os
import queue
import re
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, Field


def polygon_from_env(value: str) -> list[tuple[int, int]]:
    try: return [(int(x), int(y)) for x, y in json.loads(value)]
    except (ValueError, TypeError): return [(220, 120), (1060, 120), (1180, 700), (80, 700)]


@dataclass(slots=True)
class Settings:
    camera_id: str = os.getenv("CAMERA_ID", "border-cam-01")
    rtsp_url: str = os.getenv("RTSP_URL", "")
    # The hackathon prototype opens the built-in webcam by default. Set RTSP_URL
    # or FALLBACK_VIDEO to override it without changing code.
    fallback_video: str = os.getenv("FALLBACK_VIDEO", "0")
    # Reuse a project-local model weight when present; YOLO can still download
    # its standard weight when this file is intentionally absent.
    model_path: str = os.getenv("YOLO_MODEL", str(Path(__file__).resolve().parent.parent / "yolo11n.pt"))
    tracker: str = os.getenv("YOLO_TRACKER", "bytetrack.yaml")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
    process_fps: float = float(os.getenv("PROCESS_FPS", "8"))
    frame_buffer_size: int = int(os.getenv("FRAME_BUFFER_SIZE", "3"))
    enable_low_light: bool = os.getenv("ENABLE_LOW_LIGHT", "false").lower() == "true"
    gamma: float = float(os.getenv("GAMMA", "1.25"))
    border_polygon: list[tuple[int, int]] = field(default_factory=lambda: polygon_from_env(os.getenv("BORDER_POLYGON", "")))
    min_breach_speed_px_s: float = float(os.getenv("MIN_BREACH_SPEED_PX_S", "8"))
    watchlist_path: str = os.getenv("WATCHLIST_PATH", "watchlist.npz")
    anpr_watchlist: set[str] = field(default_factory=lambda: {x.strip().upper() for x in os.getenv("ANPR_WATCHLIST", "").split(",") if x.strip()})
    alert_webhook: str = os.getenv("ALERT_WEBHOOK", "http://127.0.0.1:8000/api/alerts")
    cooldown_seconds: float = float(os.getenv("ALERT_COOLDOWN_SECONDS", "12"))
    capture_dir: str = os.getenv("CAPTURE_DIR", str(Path.home() / "Downloads" / "IBVAP_Captures"))
    capture_cooldown_seconds: float = float(os.getenv("CAPTURE_COOLDOWN_SECONDS", "4"))


class LowLightPreprocessor:
    def __init__(self, enabled: bool, gamma: float) -> None:
        self.enabled = enabled; self.clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        inv = 1 / max(gamma, .1)
        self.lut = np.array([((i / 255) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled: return frame
        l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)); l = self.clahe.apply(l)
        return cv2.LUT(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR), self.lut)


@dataclass(slots=True)
class Detection:
    bbox: tuple[int, int, int, int]; confidence: float; class_id: int; label: str; track_id: int | None
    speed_px_s: float = 0.; vector: tuple[float, float] = (0., 0.); entered_zone: bool = False; behavior: str = "NORMAL"


class Detector:
    PERSON, VEHICLES = 0, {2, 3, 5, 7}
    def __init__(self, cfg: Settings) -> None:
        self.cfg, self.polygon = cfg, np.asarray(cfg.border_polygon, dtype=np.int32)
        self.history: dict[int, Deque[tuple[float, float, float, bool]]] = defaultdict(lambda: deque(maxlen=20))
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(cfg.model_path)
        except Exception as exc: print(f"[detector] YOLO unavailable: {exc}")
    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.model is None: return []
        try:
            result = self.model.track(frame, persist=True, tracker=self.cfg.tracker, conf=self.cfg.confidence_threshold,
                                      classes=[self.PERSON, *self.VEHICLES], verbose=False)[0]
        except Exception as exc: print(f"[detector] inference failed: {exc}"); return []
        if result.boxes is None or not len(result.boxes): return []
        ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [None] * len(result.boxes)
        output = []
        for box, conf, cls, ident in zip(result.boxes.xyxy.int().cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.int().cpu().tolist(), ids):
            d = Detection(tuple(box), float(conf), int(cls), str(result.names[cls]), ident)
            if ident is not None: d.speed_px_s, d.vector, d.entered_zone = self.update_trajectory(ident, d.bbox)
            if d.class_id == self.PERSON:
                w, h = max(d.bbox[2]-d.bbox[0],1), max(d.bbox[3]-d.bbox[1],1)
                d.behavior = "CRAWLING_DETECTED" if w/h > 1.25 else ("CLIMBING_SUSPECTED" if w/h < .28 and h > 100 else "NORMAL")
            output.append(d)
        return output
    def update_trajectory(self, ident: int, box: tuple[int,int,int,int]) -> tuple[float,tuple[float,float],bool]:
        x1,y1,x2,y2 = box; cx,cy = (x1+x2)/2,(y1+y2)/2; now=time.monotonic(); inside=cv2.pointPolygonTest(self.polygon,(cx,cy),False)>=0
        trail=self.history[ident]; speed=0.; vector=(0.,0.); entered=False
        if trail:
            px,py,pt,was_inside=trail[-1]; elapsed=max(now-pt,.001); vector=((cx-px)/elapsed,(cy-py)/elapsed); speed=math.hypot(*vector)
            entered=inside and not was_inside and speed >= self.cfg.min_breach_speed_px_s
        trail.append((cx,cy,now,inside)); return speed,vector,entered
    def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        output=frame.copy(); cv2.polylines(output,[self.polygon],True,(0,0,255),2)
        for d in detections:
            x1,y1,x2,y2=d.bbox; color=(0,0,255) if d.entered_zone or d.behavior != "NORMAL" else (0,200,0)
            cv2.rectangle(output,(x1,y1),(x2,y2),color,2); cv2.putText(output,f"{d.label} #{d.track_id or '-'} {d.confidence:.2f}",(x1,max(20,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,2)
        return output


class FaceRecognitionService:
    def __init__(self, path: str) -> None:
        self.names: list[str]=[]; self.embeddings=np.empty((0,512),np.float32); self.app=None
        if Path(path).exists():
            data=np.load(path,allow_pickle=False); self.names=data["names"].tolist(); self.embeddings=data["embeddings"].astype(np.float32)
        try:
            from insightface.app import FaceAnalysis
            self.app=FaceAnalysis(name="buffalo_l",providers=["CPUExecutionProvider"]); self.app.prepare(ctx_id=0,det_size=(320,320))
        except Exception as exc: print(f"[frs] disabled: {exc}")
    def match(self, crop: np.ndarray) -> Optional[tuple[str,float]]:
        if self.app is None or not len(self.embeddings) or not crop.size: return None
        faces=self.app.get(crop)
        if not faces:return None
        db=self.embeddings/np.maximum(np.linalg.norm(self.embeddings,axis=1,keepdims=True),1e-9); scores=db @ faces[0].normed_embedding; index=int(np.argmax(scores))
        return (self.names[index],float(scores[index])) if scores[index]>=.48 else None


class ANPRService:
    pattern=re.compile(r"^[A-Z0-9]{5,12}$")
    def __init__(self) -> None:
        self.reader=None
        try:
            import easyocr; self.reader=easyocr.Reader(["en"],gpu=False,verbose=False)
        except Exception as exc: print(f"[anpr] disabled: {exc}")
    def read_plate(self,crop:np.ndarray)->Optional[tuple[str,float]]:
        if self.reader is None or not crop.size:return None
        for _,text,conf in self.reader.readtext(crop[int(crop.shape[0]*.45):],detail=1,allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
            text=re.sub(r"[^A-Z0-9]","",text.upper())
            if self.pattern.fullmatch(text):return text,float(conf)
        return None


def thumbnail(frame: np.ndarray) -> str:
    image=cv2.resize(frame,(160,90),interpolation=cv2.INTER_AREA)
    for quality in (55,40,28,18):
        ok,data=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,quality])
        if ok and len(data)<=6000:return base64.b64encode(data).decode()
    return ""


class Publisher:
    def __init__(self, alert_url: str, cooldown: float) -> None:
        self.alert_url=alert_url; self.frame_url=alert_url.rsplit("/",1)[0]+"/frame"; self.cooldown=cooldown; self.last:dict[str,float]={}
        self.alerts:queue.Queue[dict[str,Any]]=queue.Queue(100); self.frames:queue.Queue[str]=queue.Queue(1)
        threading.Thread(target=self._alert_loop,daemon=True).start(); threading.Thread(target=self._frame_loop,daemon=True).start()
    def alert(self,camera:str,event:str,confidence:float,bbox:tuple[int,int,int,int],frame:np.ndarray,**extra:Any)->None:
        key=f"{camera}:{event}:{extra.get('track_id','')}"; now=time.monotonic()
        if now-self.last.get(key,0)<self.cooldown:return
        self.last[key]=now; payload={"timestamp":datetime.now(timezone.utc).isoformat(),"camera_id":camera,"event_type":event,"confidence":round(confidence,4),"bbox":list(bbox),"snapshot_thumbnail_b64":thumbnail(frame),**extra}
        if len(json.dumps(payload,separators=(",",":")).encode())>=10000: payload["snapshot_thumbnail_b64"]=""
        if self.alerts.full():
            try:self.alerts.get_nowait()
            except queue.Empty:pass
        self.alerts.put_nowait(payload)
    def frame(self,frame:np.ndarray)->None:
        h,w=frame.shape[:2]; preview=cv2.resize(frame,(640,max(1,int(h*640/w))),interpolation=cv2.INTER_AREA) if w>640 else frame; ok,data=cv2.imencode(".jpg",preview,[cv2.IMWRITE_JPEG_QUALITY,65])
        if not ok:return
        if self.frames.full():
            try:self.frames.get_nowait()
            except queue.Empty:pass
        self.frames.put_nowait(base64.b64encode(data).decode())
    def _alert_loop(self)->None:
        while True:
            try:requests.post(self.alert_url,json=self.alerts.get(),timeout=2.5).raise_for_status()
            except requests.RequestException as exc:print(f"[alerts] dispatch failed: {exc}")
    def _frame_loop(self)->None:
        while True:
            try:requests.post(self.frame_url,json={"image_b64":self.frames.get()},timeout=1.5).raise_for_status()
            except requests.RequestException:pass


class EvidenceStore:
    """Writes locally retained evidence only for confirmed detector observations."""
    def __init__(self, directory: str, cooldown: float) -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cooldown = cooldown
        self.last_saved: dict[str, float] = {}
    def save(self, frame: np.ndarray, detection: Detection) -> Optional[str]:
        key = f"{detection.class_id}:{detection.track_id}"
        now = time.monotonic()
        if now - self.last_saved.get(key, 0) < self.cooldown:
            return None
        self.last_saved[key] = now
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_label = re.sub(r"[^a-z0-9_-]", "_", detection.label.lower())
        destination = self.directory / f"{timestamp}_{safe_label}_track-{detection.track_id or 'new'}.jpg"
        # imwrite is atomic enough for this local hackathon workflow; failures are surfaced in the edge terminal.
        if cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            print(f"[evidence] saved {destination}")
            return str(destination)
        print(f"[evidence] failed to save {destination}")
        return None


class FrameGrabber:
    def __init__(self,source:str,size:int)->None:self.source=source;self.frames:queue.Queue[np.ndarray]=queue.Queue(size);self.running=True
    def start(self)->None:threading.Thread(target=self._run,daemon=True,name="rtsp-grabber").start()
    def _run(self)->None:
        while self.running:
            cap=self._open_capture()
            if cap is None:
                print("[ingest] camera unavailable. Check macOS Camera permission for this terminal/IDE, "
                      "close other apps using the webcam, then retrying in 3 seconds.")
                time.sleep(3);continue
            while self.running:
                ok,frame=cap.read()
                if not ok:print("[ingest] dropped stream; reconnecting");break
                if self.frames.full():
                    try:self.frames.get_nowait()
                    except queue.Empty:pass
                try:self.frames.put_nowait(frame)
                except queue.Full:pass
            cap.release();time.sleep(1)
    def _open_capture(self)->Optional[cv2.VideoCapture]:
        """Try platform-native camera backends before generic OpenCV fallback."""
        source=int(self.source) if self.source.strip().isdigit() else self.source
        if not isinstance(source,int):
            cap=cv2.VideoCapture(source,cv2.CAP_FFMPEG)
            return cap if cap.isOpened() else None
        backends=[("AVFoundation",getattr(cv2,"CAP_AVFOUNDATION",cv2.CAP_ANY)),("Auto",cv2.CAP_ANY)]
        for name,backend in backends:
            cap=cv2.VideoCapture(source,backend)
            if cap.isOpened():
                # Request a laptop-friendly resolution; drivers may choose the closest supported mode.
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
                print(f"[ingest] webcam {source} connected via {name}")
                return cap
            cap.release()
        return None
    def latest(self)->Optional[np.ndarray]:
        try:return self.frames.get(timeout=1)
        except queue.Empty:return None


def run_edge(cfg:Settings=Settings())->None:
    source=cfg.rtsp_url or cfg.fallback_video
    if not source:raise ValueError("Set RTSP_URL or FALLBACK_VIDEO.")
    grabber=FrameGrabber(source,cfg.frame_buffer_size);grabber.start();processor=LowLightPreprocessor(cfg.enable_low_light,cfg.gamma);detector=Detector(cfg);frs=FaceRecognitionService(cfg.watchlist_path);anpr=ANPRService();publisher=Publisher(cfg.alert_webhook,cfg.cooldown_seconds);evidence=EvidenceStore(cfg.capture_dir,cfg.capture_cooldown_seconds);interval=1/max(cfg.process_fps,.1);last=0.
    try:
        while True:
            frame=grabber.latest()
            if frame is None or time.monotonic()-last<interval:continue
            last=time.monotonic();frame=processor.apply(frame);detections=detector.detect(frame)
            for d in detections:
                x1,y1,x2,y2=d.bbox;subject=frame[max(0,y1):min(frame.shape[0],y2),max(0,x1):min(frame.shape[1],x2)]
                # Save an annotated evidence image for each observed person/vehicle (cooldown-limited per track).
                saved_path=evidence.save(detector.annotate(frame,[d]),d)
                evidence_meta={"evidence_file":saved_path} if saved_path else {}
                if d.entered_zone:publisher.alert(cfg.camera_id,"TRIPWIRE_BREACH",d.confidence,d.bbox,frame,track_id=d.track_id,speed_px_s=round(d.speed_px_s,2),vector=d.vector,**evidence_meta)
                if d.class_id==detector.PERSON and d.behavior=="CRAWLING_DETECTED":publisher.alert(cfg.camera_id,"CRAWLING_DETECTED",d.confidence,d.bbox,frame,track_id=d.track_id,**evidence_meta)
                if d.class_id==detector.PERSON:
                    match=frs.match(subject)
                    if match:publisher.alert(cfg.camera_id,"FRS_MATCH",match[1],d.bbox,frame,identity=match[0],track_id=d.track_id)
                elif d.class_id in detector.VEHICLES:
                    plate=anpr.read_plate(subject)
                    if plate and plate[0] in cfg.anpr_watchlist:publisher.alert(cfg.camera_id,"ANPR_ALERT",plate[1],d.bbox,frame,plate=plate[0],track_id=d.track_id)
            publisher.frame(detector.annotate(frame,detections))
    except KeyboardInterrupt:grabber.running=False


# C2 API and dashboard
app=FastAPI(title="IBVAP C2",version="0.1.0"); alert_log:deque[dict[str,Any]]=deque(maxlen=200); clients:set[WebSocket]=set(); latest_frame:bytes|None=None
class Alert(BaseModel):
    model_config=ConfigDict(extra="allow");timestamp:str;camera_id:str;event_type:str;confidence:float=Field(ge=0,le=1);bbox:list[int]=Field(min_length=4,max_length=4);snapshot_thumbnail_b64:str=""
class Frame(BaseModel):image_b64:str=Field(min_length=10,max_length=3_000_000)
async def broadcast(message:dict[str,Any])->None:
    dead=[]
    for client in clients:
        try:await client.send_json(message)
        except Exception:dead.append(client)
    for client in dead:clients.discard(client)
@app.post("/api/alerts",status_code=202)
async def receive_alert(alert:Alert)->dict[str,bool]:
    payload=alert.model_dump();alert_log.appendleft(payload);await broadcast({"type":"alert","data":payload});return {"accepted":True}
@app.get("/api/alerts")
async def list_alerts(limit:int=50)->list[dict[str,Any]]:
    if not 1<=limit<=200:raise HTTPException(422,"limit must be 1..200")
    return list(alert_log)[:limit]
@app.post("/api/frame",status_code=202)
async def receive_frame(frame:Frame)->dict[str,bool]:
    global latest_frame
    try:raw=base64.b64decode(frame.image_b64,validate=True)
    except ValueError as exc:raise HTTPException(422,"invalid JPEG Base64") from exc
    if not raw.startswith(b"\xff\xd8"):raise HTTPException(422,"preview must be JPEG")
    latest_frame=raw;return {"accepted":True}
@app.get("/api/live-frame")
async def live_frame()->Response:
    if latest_frame is None:raise HTTPException(404,"waiting for edge preview")
    return Response(latest_frame,media_type="image/jpeg",headers={"Cache-Control":"no-store"})
@app.websocket("/ws")
async def ws(socket:WebSocket)->None:
    await socket.accept();clients.add(socket)
    try:
        await socket.send_json({"type":"history","data":list(alert_log)})
        while True:await socket.receive_text()
    except WebSocketDisconnect:clients.discard(socket)
@app.get("/health")
async def health()->dict[str,Any]:return {"status":"ok","connected_dashboards":len(clients),"alert_buffer":len(alert_log)}
@app.get("/",response_class=HTMLResponse)
async def dashboard()->str:return DASHBOARD
DASHBOARD='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IBVAP // EDGE CONSOLE</title><style>
:root{--void:#050505;--graphite:#141416;--iron:#2a2a2e;--cloud:#e8e8ea;--ash:#8a8a8d;--ember:#a02a22;--orange:#fc6b2f}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 82% 5%,#351512 0,var(--void) 33%);color:var(--cloud);font:13px Inter,ui-sans-serif,system-ui,sans-serif;min-height:100vh}.noise{position:fixed;inset:0;pointer-events:none;opacity:.035;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='4'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")}.bar{height:72px;display:flex;align-items:center;justify-content:space-between;padding:0 clamp(20px,5vw,72px);border-bottom:1px solid #ffffff12;letter-spacing:.08em}.brand{font-size:15px;font-weight:800}.brand i{color:var(--orange);font-style:normal}.status{font-size:11px;color:#8ee09d}.wrap{max-width:1440px;margin:auto;padding:40px clamp(20px,5vw,72px)}.eyebrow{color:var(--orange);font-size:10px;letter-spacing:.18em;font-weight:800}.hero{display:flex;align-items:end;justify-content:space-between;gap:20px;margin:12px 0 30px}.hero h1{font-size:clamp(32px,5vw,68px);line-height:.93;letter-spacing:-.06em;margin:0}.hero p{color:var(--ash);max-width:310px;line-height:1.6;margin:0}.grid{display:grid;grid-template-columns:minmax(0,1.8fr) minmax(300px,.8fr);gap:18px}.card{background:#141416aa;backdrop-filter:blur(20px);border:1px solid #ffffff12;border-radius:18px;padding:18px}.cap{display:flex;justify-content:space-between;align-items:center;font-size:10px;color:var(--ash);letter-spacing:.14em;text-transform:uppercase;margin-bottom:14px}.live{position:relative;min-height:440px;display:grid;place-items:center;background:#080809;border:1px solid #ffffff0d;overflow:hidden}.live:before{content:'LIVE / CAM 00';position:absolute;top:14px;left:14px;z-index:1;background:#a02a22dd;padding:6px 8px;font-size:10px;font-weight:bold;letter-spacing:.12em}.live img,.live video{max-width:100%;max-height:540px;display:none}.empty{color:var(--ash);letter-spacing:.08em}.statrow{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:18px}.stat{border:1px solid #ffffff10;padding:13px}.stat b{display:block;font-size:21px}.stat span{font-size:9px;color:var(--ash);letter-spacing:.1em}.incident{position:relative;border-left:2px solid var(--ember);margin:10px 0;padding:12px;background:#ffffff06;min-height:70px}.badge{font-size:9px;letter-spacing:.08em;color:#fff;background:var(--ember);padding:4px 6px}.meta{color:var(--ash);font-size:11px;margin-top:7px}.thumb{width:94px;max-height:62px;object-fit:cover;float:right;margin-left:10px}@media(max-width:820px){.grid{grid-template-columns:1fr}.live{min-height:260px}.hero{display:block}.hero p{margin-top:16px}}</style></head><body><div class="noise"></div><header class="bar"><div class="brand">IBVAP <i>//</i> EDGE CONSOLE</div><div id="status" class="status">● CONNECTING</div></header><main class="wrap"><div class="eyebrow">INTELLIGENT BORDER VIDEO ANALYTICS</div><section class="hero"><h1>OBSERVE.<br>DETECT.<br><span style="color:var(--orange)">RESPOND.</span></h1><p>Hackathon prototype · Direct laptop camera · Real-time tracking and incident telemetry.</p></section><section class="grid"><div><div class="card"><div class="cap"><span>Live camera feed</span><span>CAMERA 00 / DIRECT</span></div><div class="live"><span id="empty" class="empty">CONNECTING TO LAPTOP CAMERA…</span><video id="camera" autoplay muted playsinline aria-label="Live laptop camera"></video><img id="feed" alt="Edge annotated camera feed"></div></div><div class="statrow"><div class="stat"><b id="count">0</b><span>INCIDENTS</span></div><div class="stat"><b id="edge">DIRECT</b><span>CAMERA SOURCE</span></div><div class="stat"><b>YOLO</b><span>VISION MODEL</span></div></div></div><aside class="card"><div class="cap"><span>Incident feed</span><span>REAL-TIME</span></div><div id="log"><p class="meta">No incidents received.</p></div></aside></section></main><script>const l=document.querySelector('#log'),s=document.querySelector('#status'),f=document.querySelector('#feed'),v=document.querySelector('#camera'),e=document.querySelector('#empty'),c=document.querySelector('#count'),q=document.querySelector('#edge');let n=0,direct=false;function add(a){if(!n)l.innerHTML='';n++;c.textContent=n;let x=document.createElement('article');x.className='incident';let i=a.snapshot_thumbnail_b64?`<img class="thumb" src="data:image/jpeg;base64,${a.snapshot_thumbnail_b64}">`:'';x.innerHTML=`${i}<span class="badge">${a.event_type.replaceAll('_',' ')}</span><div class="meta">${a.camera_id} · ${(a.confidence*100).toFixed(1)}% confidence</div><div class="meta">${new Date(a.timestamp).toLocaleTimeString()} · bbox [${a.bbox}]</div>`;l.prepend(x)}async function startCamera(){try{let stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});v.srcObject=stream;v.style.display='block';e.style.display='none';direct=true;s.textContent='● CAMERA DIRECT / LIVE'}catch(err){q.textContent='EDGE';e.textContent='CAMERA PERMISSION REQUIRED — USE BROWSER PROMPT';setInterval(()=>{f.src='/api/live-frame?'+Date.now();f.onload=()=>{f.style.display='block';e.style.display='none'}},700)}}startCamera();let w=new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+'/ws');w.onopen=()=>{if(!direct)s.textContent='● SYSTEM LIVE'};w.onmessage=z=>{let m=JSON.parse(z.data);if(m.type==='history')m.data.reverse().forEach(add);else add(m.data)};w.onclose=()=>{s.textContent='● RECONNECTING'};</script></body></html>'''

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "edge": run_edge()
    else: print("Use `uvicorn ibvap:app --host 0.0.0.0 --port 8000` for C2 or `python ibvap.py edge` for edge inference.")
