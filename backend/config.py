"""Configuration for the Animal Tracking System (runtime-editable)."""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env")

CONFIG_JSON = Path(__file__).parent / "config.json"
_BACKEND = Path(__file__).parent

# Schema: key -> (default, cast). Overrides: config.json > env (KEY_NAME) > default.
_SCHEMA = {
    "video_path": ("input_videos/video.mp4", str),
    "yolo_model": ("best.pt", str),
    "yolo_confidence": ("0.4", float),
    "tracker_config": ("bytetrack.yaml", str),
    "stream_fps": ("24", int),
    "loop_video": ("true", lambda v: v if isinstance(v, bool) else str(v).lower() == "true"),
    "event_start_threshold_s": ("1.5", float),
    "event_end_threshold_s": ("2.5", float),
    "auto_pause_minutes": ("10", int),
    "openrouter_api_key": ("", str),
    "openrouter_base_url": ("https://openrouter.ai/api/v1", str),
    "thumbnails_dir": ("thumbnails", str),
    "database_path": ("animal_tracker.db", str),
    "host": ("0.0.0.0", str),
    "port": ("8000", int),
}


class Config:
    """Mutable config. Values: config.json > env > defaults."""

    def __init__(self):
        self._overrides = {}
        if CONFIG_JSON.exists():
            try:
                self._overrides = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
            except Exception:
                pass

    def _get(self, key: str):
        default, cast = _SCHEMA[key]
        v = self._overrides.get(key)
        if v is not None and (v != "" if isinstance(v, str) else True):
            return cast(v)
        v = os.environ.get(key.upper().replace("-", "_"), default)
        return cast(v) if v != "" else cast(default)

    def __getattr__(self, key: str):
        if key.startswith("_"):
            raise AttributeError(key)
        if key not in _SCHEMA:
            raise AttributeError(key)
        v = self._get(key)
        if key == "yolo_confidence":
            return max(0.0, min(1.0, float(v)))
        return v

    @property
    def input_videos_dir(self) -> Path:
        return _BACKEND / "input_videos"

    @property
    def video_path(self) -> str:
        raw = self._get("video_path")
        p = Path(raw)
        return str(_BACKEND / raw) if not p.is_absolute() else raw

    @property
    def api_base_url(self) -> str:
        return os.environ.get("API_BASE_URL", f"http://127.0.0.1:{self.port}")

    @property
    def vlm_model(self) -> str:
        return "qwen/qwen-2.5-vl-7b-instruct"

    @property
    def llm_model(self) -> str:
        return "google/gemini-3-flash-preview:online"

    def update(self, data: dict) -> None:
        self._overrides.update(data)
        CONFIG_JSON.write_text(json.dumps(self._overrides, indent=2), encoding="utf-8")

    def video_path_relative(self) -> str:
        raw = self._get("video_path")
        p = Path(raw)
        if p.is_absolute():
            try:
                return str(p.relative_to(_BACKEND))
            except ValueError:
                return raw
        return raw

    def for_ui(self) -> dict:
        return {
            "video_path": self.video_path_relative(),
            "yolo_model": self.yolo_model,
            "yolo_confidence": self.yolo_confidence,
            "stream_fps": self.stream_fps,
            "loop_video": self.loop_video,
            "event_start_threshold_s": self.event_start_threshold_s,
            "event_end_threshold_s": self.event_end_threshold_s,
            "auto_pause_minutes": self.auto_pause_minutes,
        }


config = Config()


def save_config(data: dict) -> None:
    config.update(data)


def get_config_for_ui() -> dict:
    return config.for_ui()
