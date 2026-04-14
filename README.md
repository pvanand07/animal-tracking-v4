---
title: Animal Tracking V4
emoji: 🦁
colorFrom: yellow
colorTo: yellow
sdk: docker
pinned: false
---

# 🦁 Animal Tracker

AI-powered wildlife monitoring system with real-time detection, species identification, and interactive querying.

### Key Highlights

- **Real-time Object Tracking** — YOLOv8 with BoTSORT tracker processes video streams, assigns persistent tracking IDs, and annotates frames with bounding boxes streamed via MJPEG and WebSocket
- **VLM-based Species Identification** — Crops the best frame per tracked object and sends it to Qwen 2.5 VL 7B via OpenRouter with structured JSON schema output (common name, scientific name, description)
- **LLM-powered Data Enrichment** — When a new species is detected, calls Gemini Flash with online search to populate 18+ fields (habitat, diet, conservation status, safety info, etc.) into a SQLite species encyclopedia
- **LangChain ReAct Agent** — Interactive chat agent (`POST /api/chat`) with OpenRouter LLM that can inspect the database schema, run read-only SQL queries, and answer natural-language questions about detections, events, and animals
- **Event Lifecycle Management** — Threshold-based tracking (start after 1.5s continuous presence, end after 2.5s absence) with automatic thumbnail saving, AI identification triggers, and real-time WebSocket updates to the frontend
- **Full-stack Dashboard** — Vue.js 3 single-page app with live MJPEG video stream, real-time event feed, detection cards with thumbnails and animal details, settings panel, and an AI chat interface
- **Docker-deployed** — Containerized with FastAPI + Uvicorn, TensorRT/Jetson support, configurable pipeline (FPS, confidence, inference interval, model selection), and runtime-editable config persisted to `config.json`
- **Agent API** — Exposes database schema and read-only SQL execution (`GET /api/agent/schema`, `POST /api/agent/query`) for external AI agents to query detections, events, and animal data programmatically

## Architecture

```
video.mp4 → YOLO BoTSORT Tracker → Event Manager → AI Module → Sync → Database
                 │                       │              │
                 ▼                       ▼              ▼
           MJPEG Stream           WebSocket Events   REST API
           (annotated frames)     (live updates)     (detections/animals)
                 │                       │              │
                 └───────────── Vue.js Frontend ────────┘
```

### Pipeline Flow

1. **YOLO Tracker** — Reads `video.mp4`, runs YOLOv8 with BoTSORT tracking. Each tracked object gets a deterministic `tracking_id` (UUID5). Annotated frames are streamed via MJPEG.

2. **Event Manager** — Monitors tracking duration:
   - After **0.5s** of continuous tracking → starts an event, sends best crop to AI
   - After **2.5s** of absence → ends the event

3. **AI Module (VLM)** — Sends cropped frame to `qwen/qwen-2.5-vl-7b-instruct` via OpenRouter for species identification (common name, scientific name, description).

4. **Sync Module** — Looks up the identified animal in `animals` table by name. If not found, calls `google/gemini-3-flash-preview:online` (with web search) to gather comprehensive animal info and inserts it.

5. **Database** — SQLite with 4 tables:
   - `animals` — species encyclopedia (auto-increment ID + 17 info columns)
   - `events` — tracking event lifecycle (start/end timestamps, bounding box)
   - `ai_detections` — VLM identification results per tracking_id
   - `detections` — links tracking_id → animal_id

### Endpoints

| Endpoint | Type | Description |
|---|---|---|
| `GET /` | HTML | Vue.js dashboard |
| `GET /stream` | MJPEG | Annotated video stream |
| `WS /ws/frames` | WebSocket | Binary JPEG frames |
| `WS /ws/events` | WebSocket | Real-time event updates |
| `GET /api/detections` | REST | All detections with animal info |
| `GET /api/animals` | REST | Animals database |
| `GET /api/events` | REST | Event history |
| `GET /api/events/active` | REST | Currently active events |
| `GET /api/status` | REST | System status (FPS, tracks, etc.) |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set OpenRouter API key
export OPENROUTER_API_KEY="sk-or-..."

# 3. Place your video
cp /path/to/wildlife_footage.mp4 backend/video.mp4

# 4. Run
chmod +x run.sh && ./run.sh
# or directly:
cd backend && python main.py
```

Open **http://localhost:8000** in your browser.

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `VIDEO_PATH` | `video.mp4` | Input video file |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model (auto-downloads) |
| `YOLO_CONFIDENCE` | `0.35` | Detection confidence threshold |
| `STREAM_FPS` | `24` | Max stream frame rate |
| `LOOP_VIDEO` | `true` | Loop video when finished |
| `OPENROUTER_API_KEY` | — | OpenRouter API key (required for AI) |
| `EVENT_START_THRESHOLD_S` | `0.5` | Seconds before event triggers |
| `EVENT_END_THRESHOLD_S` | `2.5` | Seconds of absence to end event |
| `PORT` | `8000` | Server port |

## Project Structure

```
animal-tracker/
├── backend/
│   ├── main.py            # FastAPI app, WebSocket, REST API
│   ├── tracker.py          # YOLO BoTSORT video processing
│   ├── event_manager.py    # Tracking lifecycle management
│   ├── ai_module.py        # VLM identification + LLM info gathering
│   ├── sync_module.py      # Animal DB sync logic
│   ├── database.py         # SQLite schema + CRUD
│   └── config.py           # Configuration constants
├── frontend/
│   └── index.html          # Vue.js 3 CDN dashboard
├── requirements.txt
├── run.sh
└── README.md
```
