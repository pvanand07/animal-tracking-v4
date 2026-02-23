"""
Recording Coordinator — vid_id state machine.

Receives YOLO events from EventManager and manages the lifecycle of
EventRecorder instances, collapsing nearby events into a single clip.

State per active vid_id:
    IDLE → RECORDING → COOLDOWN → IDLE
                  ↑         │
                  └─────────┘  (new event during cooldown → extend)
"""
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from config import config
from event_recorder import EventRecorder
from ring_buffer import RingBuffer

log = logging.getLogger("recording_coordinator")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class RecordingCoordinator:
    """
    Central coordinator for event-triggered recordings.

    Call on_event(tracking_id, bbox) from EventManager.
    Call push_frame(frame) from Tracker for every raw capture frame.
    """

    def __init__(self, ring_buffer: RingBuffer):
        self._ring = ring_buffer
        self._lock = threading.Lock()

        # Active recording state
        self._vid_id: Optional[str] = None
        self._recorder: Optional[EventRecorder] = None
        self._recording_started_wall: Optional[float] = None

        # Timers
        self._cooldown_timer: Optional[threading.Timer] = None
        self._max_clip_timer: Optional[threading.Timer] = None

        # Camera geometry — filled on first push_frame
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._fps: float = config.stream_fps

        # Lazy import to avoid circular deps at module load time
        self._db_ready = False

    # ------------------------------------------------------------------
    # Called by Tracker._run for every captured frame
    # ------------------------------------------------------------------

    def push_frame(self, frame: np.ndarray) -> None:
        """Forward live frame to the active recorder (non-blocking)."""
        # Capture geometry from first frame
        if self._width is None and frame is not None:
            h, w = frame.shape[:2]
            with self._lock:
                self._width, self._height = w, h
            log.info("Coordinator geometry set from first frame: %dx%d", w, h)

        with self._lock:
            rec = self._recorder
        if rec is not None:
            rec.push_frame(frame)

    # ------------------------------------------------------------------
    # Called by EventManager when an event starts
    # ------------------------------------------------------------------

    def on_event(self, tracking_id: str, bbox: list) -> None:
        """
        Handle a new detection event.  Either starts a fresh recording or
        extends the cooldown window on an existing one.
        """
        log.info("on_event called: tracking_id=%s, recording_enabled=%s", tracking_id, config.recording_enabled)
        if not config.recording_enabled:
            log.info("Recording disabled (recording_enabled=false) — skipping. Set recording_enabled=true in config to enable.")
            return

        with self._lock:
            if self._vid_id is None:
                # No active recording — start one
                log.info("No active vid_id — starting new recording for tracking_id=%s", tracking_id)
                self._start_recording(tracking_id)
            else:
                # Active recording exists — link this event and reset cooldown
                log.info(
                    "Event %s linked to active vid_id %s (cooldown reset)", tracking_id, self._vid_id
                )
                self._link_event(tracking_id, self._vid_id)
                self._reset_cooldown()

    # ------------------------------------------------------------------
    # Internal helpers (must be called with self._lock held)
    # ------------------------------------------------------------------

    def _start_recording(self, tracking_id: str) -> None:
        vid_id = uuid.uuid4().hex
        self._vid_id = vid_id

        preroll = self._ring.snapshot()
        fps = self._fps
        w = self._width or 1280
        h = self._height or 720

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        recordings_dir = Path(config.recordings_dir)
        recordings_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(recordings_dir / f"{vid_id}_{ts_str}.mp4")

        log.info(
            "_start_recording: vid_id=%s preroll_frames=%d ring_maxlen=%d fps=%.1f size=%dx%d output=%s",
            vid_id, len(preroll), self._ring.maxlen, fps, w, h, output_path,
        )

        if len(preroll) == 0:
            log.warning("Ring buffer is empty — no preroll frames available. "
                        "Frames may not have been pushed yet or preroll_seconds is very short.")

        rec = EventRecorder(
            vid_id=vid_id,
            output_path=output_path,
            width=w,
            height=h,
            fps=fps,
            config_ini_path=str(Path(__file__).parent / "config.ini"),
        )
        self._recorder = rec
        self._recording_started_wall = time.time()

        # Persist to DB outside the lock to avoid blocking, but start recorder now
        threading.Thread(
            target=self._db_create_recording,
            args=(vid_id, output_path, tracking_id, len(preroll) / max(fps, 1)),
            daemon=True,
        ).start()

        rec.start(preroll)

        self._reset_cooldown()
        self._start_max_clip_timer()

        log.info(
            "Recording started: vid_id=%s, tracking_id=%s, cooldown=%.1fs, max_clip=%ds, output=%s",
            vid_id, tracking_id, config.cooldown_seconds, config.max_clip_seconds, output_path,
        )

    def _reset_cooldown(self) -> None:
        """(Re)start the cooldown timer — must hold self._lock."""
        if self._cooldown_timer is not None:
            self._cooldown_timer.cancel()
        delay = config.cooldown_seconds
        self._cooldown_timer = threading.Timer(delay, self._on_cooldown_expired)
        self._cooldown_timer.daemon = True
        self._cooldown_timer.start()

    def _start_max_clip_timer(self) -> None:
        if self._max_clip_timer is not None:
            self._max_clip_timer.cancel()
        cap = config.max_clip_seconds
        self._max_clip_timer = threading.Timer(cap, self._on_max_clip_expired)
        self._max_clip_timer.daemon = True
        self._max_clip_timer.start()

    def _link_event(self, tracking_id: str, vid_id: str) -> None:
        threading.Thread(
            target=self._db_link_event,
            args=(tracking_id, vid_id),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Timer callbacks (executed in Timer threads — no lock held)
    # ------------------------------------------------------------------

    def _on_cooldown_expired(self) -> None:
        log.info("Cooldown expired — stopping recording for vid_id=%s", self._vid_id)
        self._finalise()

    def _on_max_clip_expired(self) -> None:
        log.info(
            "Max clip duration reached — stopping recording for vid_id=%s", self._vid_id
        )
        self._finalise()

    def _finalise(self) -> None:
        with self._lock:
            rec = self._recorder
            vid_id = self._vid_id
            self._recorder = None
            self._vid_id = None
            self._recording_started_wall = None
            if self._cooldown_timer:
                self._cooldown_timer.cancel()
                self._cooldown_timer = None
            if self._max_clip_timer:
                self._max_clip_timer.cancel()
                self._max_clip_timer = None

        if rec is None:
            return

        # Stop in a separate thread so we don't block the timer thread long
        def _do_stop():
            rec.stop()
            duration = rec.duration_s or 0.0
            if vid_id:
                self._db_end_recording(vid_id, duration)

        threading.Thread(target=_do_stop, daemon=True).start()

    # ------------------------------------------------------------------
    # Database operations (run in background threads)
    # ------------------------------------------------------------------

    def _db_create_recording(
        self, vid_id: str, filepath: str, tracking_id: str, preroll_s: float
    ) -> None:
        try:
            from database import create_recording, link_event_to_recording
            create_recording(vid_id, filepath, preroll_s)
            link_event_to_recording(tracking_id, vid_id)
        except Exception as exc:
            log.error("DB create_recording failed: %s", exc)

    def _db_link_event(self, tracking_id: str, vid_id: str) -> None:
        try:
            from database import link_event_to_recording
            link_event_to_recording(tracking_id, vid_id)
        except Exception as exc:
            log.error("DB link_event_to_recording failed: %s", exc)

    def _db_end_recording(self, vid_id: str, duration_s: float) -> None:
        try:
            from database import end_recording
            end_recording(vid_id, _now_iso(), duration_s)
        except Exception as exc:
            log.error("DB end_recording failed: %s", exc)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recorder is not None

    @property
    def active_vid_id(self) -> Optional[str]:
        with self._lock:
            return self._vid_id
