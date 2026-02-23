---
title: Animal Tracking V4
emoji: 📊
colorFrom: yellow
colorTo: yellow
sdk: docker
pinned: false
---

# 🦁 Animal Tracker (V4)

Real-time animal detection, tracking, and identification system powered by YOLO, vision/LLM APIs (OpenRouter), and optional event-triggered recording. Supports video files, webcam, and live MJPEG/WebSocket streams with a Vue.js dashboard.

## Architecture

```
Video / Webcam → YOLO Tracker (BoTSORT) → Event Manager → AI Module (VLM) → Sync Module → Database
       │                    │                    │
       │                    ├→ Ring Buffer ──────→ Recording Coordinator → Event Recorder (GStreamer)
       │                    │
       ▼                    ▼
  MJPEG Stream         WebSocket Events
  (/stream)             (/ws/events)
       │                    │
       └──────── Vue.js Frontend (REST API) ────────┘
```

### Pipeline flow

1. **Tracker** — Reads a video file (`input_videos/video.mp4`) or webcam, runs YOLO (`.pt` or TensorRT `.engine`) with BoTSORT. Each tracked object gets a unique `tracking_id`. Annotated frames are streamed via MJPEG; raw frames are pushed to a ring buffer for preroll when recording is enabled.

2. **Event Manager** — Monitors tracking duration:
   - After **event_start_threshold_s** (default 1.5s) of continuous tracking → starts an event, saves thumbnail, sends best crop to AI.
   - After **event_end_threshold_s** (default 2.5s) of absence → ends the event.

3. **AI Module (VLM)** — Sends cropped frame to OpenRouter VLM (e.g. `google/gemini-3-flash-preview`) for species identification (common name, scientific name, description).

4. **Sync Module** — Looks up the identified animal in the `animals` table. If not found, calls an LLM with web search (e.g. `google/gemini-3-flash-preview:online`) to fetch animal info and inserts it, then creates a detection record.

5. **Recording (optional)** — When `recording_enabled` is true, `RecordingCoordinator` receives events from the Event Manager. On event start it creates an `EventRecorder`, writes preroll (from `RingBuffer`) then live frames via GStreamer to MP4. Supports preroll seconds, cooldown, and max clip length; recordings are stored and exposed via `/recordings` and `/api/recordings`.

6. **Database** — SQLite (WAL) with tables:
   - `animals` — Species encyclopedia (ID + animal info columns including safety_info, is_dangerous).
   - `events` — Tracking event lifecycle (start/end times, bbox, start_frame/end_frame, optional vid_id).
   - `ai_detections` — VLM identification per tracking_id.
   - `detections` — Links tracking_id → animal_id.
   - `recordings` — Recording sessions (vid_id, filepath, started_at, ended_at, duration_s, preroll_s).

## API endpoints

| Endpoint | Type | Description |
|----------|------|-------------|
| `GET /` | HTML | Vue.js dashboard |
| `GET /stream` | MJPEG | Annotated video stream |
| `GET /api/video/{tracking_id}` | MJPEG | Event clip (start_frame..end_frame) |
| `WS /ws/frames` | WebSocket | Binary JPEG frames |
| `WS /ws/events` | WebSocket | Real-time event updates (init + live) |
| `GET /api/detections` | REST | All detections with animal/event/recording info |
| `GET /api/detections/{tracking_id}` | REST | Single detection detail |
| `GET /api/animals` | REST | Animals database |
| `GET /api/events` | REST | Event history |
| `GET /api/events/active` | REST | Currently active events |
| `GET /api/recordings` | REST | List of recordings |
| `GET /api/status` | REST | Tracker status (running, FPS, active tracks, recording) |
| `GET /api/videos` | REST | List videos in input_videos |
| `POST /api/videos/upload` | REST | Upload video to input_videos |
| `GET /api/config` | REST | Current config for UI |
| `PATCH /api/config` | REST | Update config (persisted to config.json) |
| `POST /api/config/save` | REST | Save config and restart tracker |
| `POST /api/tracker/start` | REST | Start tracking (with optional auto-pause) |
| `POST /api/tracker/pause` | REST | Pause tracking |
| `POST /api/db/reset` | REST | Reset database (events, detections, animals, recordings) |
| `GET /api/agent/schema` | REST | DB schema (plain text: tables, columns, indexes) |
| `POST /api/agent/query` | REST | Read-only SQL (SELECT only) |
| `POST /api/chat` | REST | NDJSON stream: chat with LangChain agent (schema + SQL tools) |

Static: `/thumbnails`, `/recordings` (direct file access).

## Setup

### 1. Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment

In `backend/.env` (or root `.env`), set your OpenRouter API key:

```bash
OPENROUTER_API_KEY="sk-or-..."
```

See `backend/.env.example`.

### 3. Video source

- **Video file:** Place a video at `backend/input_videos/video.mp4`, or set `video_path` in config / `VIDEO_PATH` when using the launch script.
- **Webcam:** In Settings, enable “Use webcam” and set the device index (config: `use_webcam`, `webcam_index`).

### 4. YOLO model

- Use a `.pt` (PyTorch) or `.engine` (TensorRT) model. Default in config is `yolo26n.engine`.
- To export TensorRT from a PyTorch model (e.g. on Jetson): use `backend/convert_engine.py` (e.g. load `best.pt` and `model.export(format="engine")`).

### 5. Run

From the project root:

```bash
./run.sh
```

Or manually:

```bash
cd backend && python main.py
```

Open **http://localhost:8000** (or set `PORT` for `run.sh`). The tracker starts paused; use the dashboard to start/pause and adjust settings.

### Docker

The repo includes a `Dockerfile`. The container runs the app on port **7860** (e.g. for Hugging Face Spaces). Build and run as needed; ensure `OPENROUTER_API_KEY` is provided at runtime.

## Configuration

- **Runtime config** is stored in `backend/config.json` and can be changed via the UI or `PATCH /api/config`. Use “Save & restart” to apply tracker-affecting options.
- **Recording:** `recording_enabled`, `preroll_seconds`, `cooldown_seconds`, `max_clip_seconds`, `recordings_dir`.
- **Tracker:** `video_path`, `yolo_model`, `yolo_confidence`, `stream_fps`, `inference_imgsz`, `inference_interval`, `loop_video`, `use_webcam`, `webcam_index`, `event_start_threshold_s`, `event_end_threshold_s`, `auto_pause_minutes`.
- **GStreamer / standalone recording:** `backend/config.ini` is used by `record.py` (and by the event recorder) for camera/encoding/overlay (e.g. device, resolution, bitrate). Not required for the main app if you only use file/webcam input and do not use event recording.

## Project structure

```
animal-tracking-v4/
├── backend/
│   ├── main.py                 # FastAPI app: streams, REST, WebSocket, config, tracker control
│   ├── tracker.py              # YOLO BoTSORT video/webcam processing, ring buffer feed
│   ├── event_manager.py        # Tracking lifecycle, thumbnails, AI trigger
│   ├── ai_module.py            # VLM identification + LLM animal info (OpenRouter)
│   ├── sync_module.py          # Animal DB sync and detection creation
│   ├── database.py             # SQLite schema and CRUD
│   ├── config.py               # Config (config.json + env)
│   ├── config.ini              # GStreamer/camera/encoding (record.py, event recorder)
│   ├── recording_coordinator.py # Event-triggered recording state machine
│   ├── ring_buffer.py          # Preroll frame buffer
│   ├── event_recorder.py       # GStreamer appsrc → MP4
│   ├── ai_chat.py              # LangChain agent (schema + read-only SQL tools)
│   ├── record.py               # Standalone GStreamer camera recorder (optional)
│   ├── convert_engine.py       # YOLO .pt → TensorRT .engine export
│   ├── input_videos/           # Video files (e.g. video.mp4)
│   ├── thumbnails/             # Event thumbnails
│   ├── recordings/             # Event-triggered MP4 clips
│   └── .env / .env.example     # OPENROUTER_API_KEY
├── frontend/
│   ├── index.html              # Vue.js 3 dashboard (stream, events, detections, settings, chat)
│   ├── index_v0.html           # Alternate UI
│   └── chat.html               # Chat UI
├── requirements.txt
├── run.sh
├── Dockerfile
└── README.md
```

## Notes

- **Jetson / TensorRT:** `main.py` sets `LD_LIBRARY_PATH` for cuDNN/TensorRT so `.engine` models load correctly.
- **Agent / Chat:** The `/api/chat` endpoint streams NDJSON from a LangChain agent that can query the DB (read-only SQL) and use the schema; the dashboard can show a chat panel.
- **Recordings:** Event-triggered clips are written under `backend/recordings/` and listed via `GET /api/recordings`; events can reference a recording via `vid_id`.
