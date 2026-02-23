"""Main FastAPI application: WebSocket streams, MJPEG endpoint, REST API, thumbnail serving."""
# Set LD_LIBRARY_PATH for TensorRT/cuDNN before any ultralytics import (required for .engine on Jetson)
import os
_cudnn_paths = [
    "/usr/lib/aarch64-linux-gnu",
    "/usr/local/cuda/lib64",
    "/usr/lib",
]
_ld = os.environ.get("LD_LIBRARY_PATH", "")
_extra = ":".join(p for p in _cudnn_paths if os.path.isdir(p) and p not in _ld)
if _extra:
    os.environ["LD_LIBRARY_PATH"] = _extra + (":" + _ld if _ld else "")

print("OR KEY", os.getenv("OPENROUTER_API_KEY"))

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from config import config, get_config_for_ui, save_config
from database import (
    init_db, get_active_events, get_all_events, get_all_detections,
    get_all_animals, get_detection_by_tracking, get_detection_detail, get_event_by_tracking_id,
    get_schema_txt, execute_read_only_query, reset_db, get_all_recordings,
)
from event_manager import EventManager
from tracker import Tracker
from recording_coordinator import RecordingCoordinator
from ai_chat import process_message as chat_process_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-14s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# ── Globals ────────────────────────────────────────────────────
event_manager = EventManager()
# Coordinator is wired to the tracker's ring_buffer after tracker is created
tracker = Tracker(event_manager)
recording_coordinator = RecordingCoordinator(tracker.ring_buffer)
tracker.recording_coordinator = recording_coordinator
event_ws_clients: list[WebSocket] = []
_auto_pause_task: asyncio.Task | None = None


async def broadcast_event(data: dict):
    """Send event updates to all connected WebSocket clients."""
    msg = json.dumps(data)
    disconnected = []
    for ws in event_ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        event_ws_clients.remove(ws)


# ── Lifespan ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    os.makedirs(config.thumbnails_dir, exist_ok=True)
    os.makedirs(config.input_videos_dir, exist_ok=True)
    os.makedirs(config.recordings_dir, exist_ok=True)
    # Migrate legacy video.mp4 from backend root to input_videos
    legacy_video = Path(__file__).parent / "video.mp4"
    if legacy_video.exists():
        dest = config.input_videos_dir / "video.mp4"
        if not dest.exists():
            import shutil
            shutil.move(str(legacy_video), str(dest))
            log.info("Moved video.mp4 to input_videos/")
    log.info("Database initialized")

    if not Path(config.video_path).exists():
        log.warning(f"Video not found: {config.video_path} — tracker will wait for it")

    # Store the event loop for cross-thread communication
    loop = asyncio.get_event_loop()

    def threadsafe_event_update(data: dict):
        try:
            asyncio.run_coroutine_threadsafe(broadcast_event(data), loop)
        except Exception:
            pass

    event_manager.on_event_update = threadsafe_event_update
    event_manager.on_recording_event = recording_coordinator.on_event
    # Tracker starts paused; user clicks Start in UI (auto-pauses after 10 min)
    log.info("Tracker ready (paused — click Start to begin)")

    yield

    # Shutdown
    if _auto_pause_task and not _auto_pause_task.done():
        _auto_pause_task.cancel()
    tracker.stop()
    log.info("Shutdown complete")


app = FastAPI(title="Animal Tracker", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve thumbnails as static files
os.makedirs(config.thumbnails_dir, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=config.thumbnails_dir), name="thumbnails")

# Serve recordings as static files
os.makedirs(config.recordings_dir, exist_ok=True)
app.mount("/recordings", StaticFiles(directory=config.recordings_dir), name="recordings")


# ── MJPEG Stream ───────────────────────────────────────────────

def mjpeg_generator():
    while tracker.running or not tracker.video_finished:
        tracker.wait_for_frame(timeout=1.0)
        frame = tracker.latest_frame
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
        else:
            time.sleep(0.03)


@app.get("/stream")
def video_stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── MJPEG clip by tracking_id ───────────────────────────────────

def mjpeg_clip_generator(tracking_id: str):
    """Yield MJPEG frames for the clip range (start_frame..end_frame) of the given event."""
    event = get_event_by_tracking_id(tracking_id)
    if not event:
        return
    start_f = event.get("start_frame")
    end_f = event.get("end_frame")
    if start_f is None or end_f is None:
        return
    start_frame = int(start_f)
    end_frame = int(end_f)
    if start_frame < 0 or end_frame < start_frame:
        return
    path = Path(config.video_path)
    if not path.exists():
        return
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for fn in range(start_frame, min(end_frame + 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )
    finally:
        cap.release()


@app.get("/api/video/{tracking_id}")
def video_clip(tracking_id: str):
    """Stream the event clip as MJPEG for the given tracking_id."""
    event = get_event_by_tracking_id(tracking_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    if event.get("start_frame") is None or event.get("end_frame") is None:
        raise HTTPException(
            status_code=400,
            detail="Event has no frame range (start_frame/end_frame missing)",
        )
    if not Path(config.video_path).exists():
        raise HTTPException(status_code=503, detail="Video file not available")
    return StreamingResponse(
        mjpeg_clip_generator(tracking_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── WebSocket: Frame Stream ────────────────────────────────────

@app.websocket("/ws/frames")
async def ws_frames(websocket: WebSocket):
    await websocket.accept()
    log.info("Frame WebSocket client connected")
    try:
        while True:
            frame = tracker.latest_frame
            if frame:
                await websocket.send_bytes(frame)
            await asyncio.sleep(1.0 / 24)
    except WebSocketDisconnect:
        log.info("Frame WebSocket client disconnected")


# ── WebSocket: Event Updates ───────────────────────────────────

@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await websocket.accept()
    event_ws_clients.append(websocket)
    log.info(f"Event WebSocket client connected (total: {len(event_ws_clients)})")
    try:
        active = get_active_events()
        await websocket.send_text(json.dumps({"type": "init", "events": active}))
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in event_ws_clients:
            event_ws_clients.remove(websocket)
        log.info("Event WebSocket client disconnected")


# ── REST API ───────────────────────────────────────────────────

@app.get("/api/detections")
def api_detections(limit: int = Query(default=200, le=1000)):
    return get_all_detections(limit)


@app.get("/api/detections/{tracking_id}")
def api_detection_detail(tracking_id: str):
    det = get_detection_detail(tracking_id)
    if not det:
        return JSONResponse({"error": "not found"}, 404)
    return det


@app.get("/api/animals")
def api_animals():
    return get_all_animals()


@app.get("/api/events")
def api_events(limit: int = Query(default=100, le=500)):
    return get_all_events(limit)


@app.get("/api/events/active")
def api_active_events():
    return get_active_events()


@app.get("/api/status")
def api_status():
    return {
        "tracker_running": tracker.running,
        "video_finished": tracker.video_finished,
        "frame_count": tracker.frame_count,
        "fps": round(tracker.fps, 1),
        "active_tracks": len(event_manager.get_active_tracks()),
        "active_tracks_list": event_manager.get_active_tracks(),
        "recording": recording_coordinator.is_recording,
        "active_vid_id": recording_coordinator.active_vid_id,
    }


# ── Recordings API ───────────────────────────────────────────────

@app.get("/api/recordings")
def api_recordings(limit: int = Query(default=50, le=200)):
    """Return a list of completed and in-progress event recordings."""
    return get_all_recordings(limit)


# ── Videos API ───────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@app.get("/api/videos")
def api_list_videos():
    """List video files in input_videos folder."""
    videos_dir = config.input_videos_dir
    if not videos_dir.exists():
        return []
    videos = []
    for f in sorted(videos_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            rel_path = f"input_videos/{f.name}"
            videos.append({"name": f.name, "path": rel_path})
    return videos


@app.post("/api/videos/upload")
async def api_upload_video(file: UploadFile = File(...)):
    """Upload a video file to input_videos folder."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format. Use: {', '.join(VIDEO_EXTENSIONS)}")
    target = config.input_videos_dir / (file.filename or "upload.mp4")
    try:
        content = await file.read()
        with open(target, "wb") as f:
            f.write(content)
        rel_path = f"input_videos/{target.name}"
        save_config({"video_path": rel_path})
        log.info("Uploaded video: %s", target.name)
        return {"ok": True, "path": rel_path, "name": target.name}
    except Exception as e:
        log.error("Video upload failed: %s", e)
        raise HTTPException(500, str(e))


# ── Config API ───────────────────────────────────────────────────

@app.get("/api/config")
def api_get_config():
    """Return current config for settings UI."""
    return get_config_for_ui()


class ConfigUpdate(BaseModel):
    video_path: Optional[str] = None
    yolo_model: Optional[str] = None
    yolo_confidence: Optional[float] = None
    stream_fps: Optional[int] = None
    inference_imgsz: Optional[int] = None
    inference_half: Optional[bool] = None
    inference_interval: Optional[int] = None
    loop_video: Optional[bool] = None
    event_start_threshold_s: Optional[float] = None
    event_end_threshold_s: Optional[float] = None
    auto_pause_minutes: Optional[int] = None
    use_webcam: Optional[bool] = None
    webcam_index: Optional[int] = None
    recording_enabled: Optional[bool] = None
    preroll_seconds: Optional[float] = None
    cooldown_seconds: Optional[float] = None
    max_clip_seconds: Optional[int] = None


@app.patch("/api/config")
def api_patch_config(body: ConfigUpdate = Body(...)):
    """Update config (persists to config.json). Does not restart tracker."""
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    if not data:
        return {"ok": True, "message": "No changes"}
    save_config(data)
    log.info("Config updated: %s", list(data.keys()))
    return {"ok": True, "message": "Settings saved."}


@app.post("/api/config/save")
async def api_config_save(body: ConfigUpdate = Body(...)):
    """Save config and restart tracker so changes take effect."""
    global _auto_pause_task
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    if not data:
        return {"ok": True, "message": "No changes"}
    save_config(data)
    log.info("Config saved: %s", list(data.keys()))
    was_running = tracker.running
    if _auto_pause_task and not _auto_pause_task.done():
        _auto_pause_task.cancel()
        _auto_pause_task = None
    tracker.stop()
    if was_running:
        tracker.start()
        cfg = get_config_for_ui()
        mins = cfg.get("auto_pause_minutes", 10)
        delay = mins * 60

        async def _auto_pause():
            await asyncio.sleep(delay)
            tracker.stop()
            log.info("Tracker auto-paused after %d minutes", mins)

        _auto_pause_task = asyncio.create_task(_auto_pause())
        return {"ok": True, "message": f"Settings saved. Tracker restarted (auto-pause in {mins} min)."}
    return {"ok": True, "message": "Settings saved."}


@app.post("/api/db/reset")
def api_db_reset():
    """Reset database: delete all events, detections, ai_detections, and animals."""
    reset_db()
    log.info("Database reset by user")
    return {"ok": True, "message": "Database reset."}


# ── Tracker Control API ──────────────────────────────────────────

@app.post("/api/tracker/start")
async def api_tracker_start():
    """Start tracking. Auto-pauses after configured minutes (default 10)."""
    global _auto_pause_task
    if tracker.running:
        return {"ok": True, "message": "Already running"}
    tracker.start()
    log.info("Tracker started by user")
    # Schedule auto-pause
    cfg = get_config_for_ui()
    mins = cfg.get("auto_pause_minutes", 10)
    delay = mins * 60

    async def _auto_pause():
        await asyncio.sleep(delay)
        tracker.stop()
        log.info("Tracker auto-paused after %d minutes", mins)

    _auto_pause_task = asyncio.create_task(_auto_pause())
    return {"ok": True, "message": f"Tracking started. Will auto-pause after {mins} minutes."}


@app.post("/api/tracker/pause")
async def api_tracker_pause():
    """Pause tracking."""
    global _auto_pause_task
    if _auto_pause_task and not _auto_pause_task.done():
        _auto_pause_task.cancel()
        _auto_pause_task = None
    tracker.stop()
    log.info("Tracker paused by user")
    return {"ok": True, "message": "Tracking paused."}


# ── Agent API (schema, docs, read-only SQL) ──────────────────────

AGENT_DOCS = """
# Animal Tracker API – Agent reference

## Schema & data
GET /api/agent/schema  →  Database schema and this doc as plain text (Content-Type: text/plain).

## Read-only SQL
POST /api/agent/query
Body: {"query": "SELECT ..."}
Only SELECT is allowed. Returns JSON: {"ok": true, "rows": [...]} or {"ok": false, "error": "..."}.

## REST endpoints
GET  /api/detections?limit=200     →  All detections (joined with animals, events, ai_detections)
GET  /api/detections/{tracking_id} →  Single detection by tracking_id
GET  /api/animals                  →  All animals
GET  /api/events?limit=100         →  All events (with ai_detections)
GET  /api/events/active            →  Active events (end_time IS NULL)
GET  /api/status                   →  Tracker status (running, fps, active_tracks, etc.)
GET  /api/video/{tracking_id}      →  MJPEG clip for that event (start_frame..end_frame)

## Streams
GET  /stream       →  MJPEG video stream
WS   /ws/frames    →  Raw JPEG frames
WS   /ws/events    →  Live event updates (JSON); send "ping" for "pong"
"""


@app.get("/api/agent/schema", response_class=PlainTextResponse)
def api_agent_schema():
    """Return database schema and API docs as plain text for AI agents."""
    schema = get_schema_txt()
    return schema + "\n\n" + AGENT_DOCS.strip()


class QueryRequest(BaseModel):
    query: str


@app.post("/api/agent/query")
def api_agent_query(body: QueryRequest = Body(...)):
    """Execute a read-only SQL query (SELECT only). For AI agent use."""
    rows, err = execute_read_only_query(body.query.strip())
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    return {"ok": True, "rows": rows}


# ── Agent chat (LangChain + OpenRouter) ──────────────────────────

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    model_id: Optional[str] = None


async def chat_stream(request: ChatRequest):
    """Stream agent events as NDJSON."""
    async for event in chat_process_message(
        request.message,
        thread_id=request.thread_id,
        model_id=request.model_id,
    ):
        yield json.dumps(event, ensure_ascii=False) + "\n"


@app.post("/api/chat")
async def api_chat(body: ChatRequest = Body(...)):
    """Stream chat with LangChain agent (OpenRouter + schema + SQL tools)."""
    return StreamingResponse(
        chat_stream(body),
        media_type="application/x-ndjson",
    )


# ── Frontend ───────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    return HTMLResponse(index.read_text(encoding="utf-8"))


# ── Run ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.host, port=config.port, reload=False)
