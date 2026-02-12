#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# Animal Tracker — Launch Script
# ─────────────────────────────────────────────────────────
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/backend"

# Check for video
if [ ! -f "${VIDEO_PATH:-video.mp4}" ] && [ ! -f "input_videos/video.mp4" ]; then
    echo "⚠  No video.mp4 found. Place a video file at:"
    echo "   $DIR/backend/input_videos/video.mp4"
    echo "   or set VIDEO_PATH=/path/to/video.mp4"
    echo ""
fi

# Install deps if needed
if ! python -c "import ultralytics" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r "$DIR/requirements.txt" --break-system-packages -q
fi

echo "🚀 Starting Animal Tracker on http://localhost:${PORT:-8000}"
echo "   MJPEG stream:  http://localhost:${PORT:-8000}/stream"
echo "   WebSocket:     ws://localhost:${PORT:-8000}/ws/events"
echo "   API:           http://localhost:${PORT:-8000}/api/detections"
echo ""

python main.py
