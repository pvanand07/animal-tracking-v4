"""Event Manager: tracks object lifecycles, saves thumbnails, triggers AI identification."""
import os
import time
import uuid
import logging
import threading
import cv2
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from config import config
from database import create_event, end_event, update_event_last_seen, upsert_ai_detection, delete_tracker_entry, now_iso
from ai_module import identify_animal
from sync_module import sync_detection

log = logging.getLogger("event_manager")

NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

# Ensure thumbnails directory exists
os.makedirs(config.thumbnails_dir, exist_ok=True)


# def make_tracking_id(track_id: int) -> str:
#     """Deterministic tracking_id from YOLO track integer."""
#     return f"tracking_{uuid.uuid5(NAMESPACE, str(track_id))}"

def make_tracking_id(track_id: int) -> str:
    """Generate a unique tracking_id using random UUID4."""
    return f"{track_id}_{uuid.uuid4().hex[:5]}"


@dataclass
class TrackedObject:
    track_id: int
    tracking_id: str
    first_seen: float           # monotonic clock for duration math
    last_seen: float            # monotonic clock for absence math
    last_seen_dt: str = ""      # ISO datetime string for DB
    bbox: list = field(default_factory=list)
    best_crop: np.ndarray | None = None
    best_crop_area: int = 0
    event_started: bool = False
    ai_sent: bool = False
    ai_done: bool = False
    ended: bool = False
    thumbnail_saved: bool = False


class EventManager:
    """
    Manages the lifecycle of tracked objects:
    - After EVENT_START_THRESHOLD_S of tracking → start event, save thumbnail, send crop to AI
    - After EVENT_END_THRESHOLD_S of absence → end event
    - Continuously updates last_seen datetime
    """

    def __init__(self):
        self._tracks: dict[int, TrackedObject] = {}
        self._lock = threading.Lock()
        self.on_event_update = None     # callback: (event_dict) -> None
        self.on_recording_event = None  # callback: (tracking_id: str, bbox: list) -> None

    def update(self, detections: list[dict], frame: np.ndarray, mono_time: float, frame_number: int | None = None, skip_crop_update: bool = False):
        """
        Called each frame with list of detections:
        [{"track_id": int, "bbox": [x1,y1,x2,y2], "class_name": str, "confidence": float}, ...]
        mono_time: time.monotonic() value for duration/absence calculations.
        frame_number: video frame index (0-based) for this frame.
        skip_crop_update: if True, only update last_seen (no crop/thumbnail updates); use when reusing previous detections.
        """
        now = mono_time
        now_dt = now_iso()
        seen_ids = set()

        for det in detections:
            tid = det["track_id"]
            seen_ids.add(tid)
            bbox = det["bbox"]

            with self._lock:
                if tid not in self._tracks:
                    tracking_id = make_tracking_id(tid)
                    self._tracks[tid] = TrackedObject(
                        track_id=tid,
                        tracking_id=tracking_id,
                        first_seen=now,
                        last_seen=now,
                        last_seen_dt=now_dt,
                        bbox=bbox,
                    )
                    log.debug(f"New track: {tid} → {tracking_id}")
                else:
                    obj = self._tracks[tid]
                    obj.last_seen = now
                    obj.last_seen_dt = now_dt
                    obj.bbox = bbox

                obj = self._tracks[tid]

                # Keep the largest crop for best AI identification + thumbnail (skip when reusing detections from another frame)
                if not skip_crop_update:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    area = (x2 - x1) * (y2 - y1)
                    if frame is not None and area > obj.best_crop_area:
                        h, w = frame.shape[:2]
                        pad = 20
                        cx1 = max(0, x1 - pad)
                        cy1 = max(0, y1 - pad)
                        cx2 = min(w, x2 + pad)
                        cy2 = min(h, y2 + pad)
                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            obj.best_crop = crop.copy()
                            obj.best_crop_area = area
                            # Re-save thumbnail with better crop
                            if obj.event_started:
                                self._save_thumbnail(obj)

                    # Initial crop capture for new tracks
                    if obj.best_crop is None and frame is not None:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        area = (x2 - x1) * (y2 - y1)
                        h, w = frame.shape[:2]
                        pad = 20
                        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            obj.best_crop = crop.copy()
                            obj.best_crop_area = area

        # Check thresholds
        with self._lock:
            for tid, obj in list(self._tracks.items()):
                duration = obj.last_seen - obj.first_seen
                absence = now - obj.last_seen

                # Start event after threshold
                if not obj.event_started and duration >= config.event_start_threshold_s:
                    obj.event_started = True
                    create_event(obj.tracking_id, obj.bbox, start_frame=frame_number)
                    log.info(f"Event started: {obj.tracking_id} (tracked {duration:.1f}s)")

                    # Notify recording coordinator (preroll + live recording trigger)
                    if self.on_recording_event:
                        log.info("Firing on_recording_event for tracking_id=%s", obj.tracking_id)
                        try:
                            self.on_recording_event(obj.tracking_id, obj.bbox)
                        except Exception as exc:
                            log.error("on_recording_event callback error: %s", exc)
                    else:
                        log.debug("on_recording_event not wired (no coordinator attached)")

                    # Save thumbnail
                    self._save_thumbnail(obj)

                    self._notify_event_update(obj, "started")

                    # Send to AI for identification
                    if not obj.ai_sent and obj.best_crop is not None:
                        obj.ai_sent = True
                        t = threading.Thread(
                            target=self._run_ai_identification,
                            args=(obj.tracking_id, obj.best_crop.copy()),
                            daemon=True,
                        )
                        t.start()

                # Update last_seen in DB for active events (every ~1s to reduce writes)
                if obj.event_started and not obj.ended and tid in seen_ids:
                    # Throttle DB writes: update every ~30 frames
                    if int(now * 10) % 10 == 0:
                        update_event_last_seen(obj.tracking_id)

                # End event after absence threshold
                if obj.event_started and not obj.ended and absence >= config.event_end_threshold_s:
                    obj.ended = True
                    end_event(obj.tracking_id, end_frame=frame_number)
                    log.info(f"Event ended: {obj.tracking_id} (absent {absence:.1f}s)")
                    self._notify_event_update(obj, "ended")

                # Cleanup old ended tracks
                if obj.ended and absence > 10.0:
                    del self._tracks[tid]

    def _save_thumbnail(self, obj: TrackedObject):
        """Save cropped thumbnail as tracking_id.jpg."""
        if obj.best_crop is None:
            return
        try:
            path = os.path.join(config.thumbnails_dir, f"{obj.tracking_id}.jpg")
            cv2.imwrite(path, obj.best_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            obj.thumbnail_saved = True
            log.debug(f"Thumbnail saved: {path}")
        except Exception as e:
            log.error(f"Failed to save thumbnail for {obj.tracking_id}: {e}")

    def _run_ai_identification(self, tracking_id: str, crop: np.ndarray):
        """Run VLM identification + sync in a background thread."""
        try:
            result = identify_animal(crop)
            if result and result.get("common_name"):
                upsert_ai_detection(
                    tracking_id,
                    result["common_name"],
                    result.get("scientific_name", ""),
                    result.get("description", ""),
                )
                sync_result = sync_detection(tracking_id)
                if sync_result:
                    log.info(f"AI + Sync complete for {tracking_id}: {result['common_name']}")
                    self._notify_event_update_by_id(tracking_id, "identified", result)
            else:
                log.info(f"AI could not identify animal for {tracking_id} — deleting tracker entry")
                delete_tracker_entry(tracking_id)
        except Exception as e:
            log.error(f"AI identification error for {tracking_id}: {e}")

    def _notify_event_update(self, obj: TrackedObject, status: str):
        if self.on_event_update:
            self.on_event_update({
                "tracking_id": obj.tracking_id,
                "status": status,
                "bbox": obj.bbox,
                "track_id": obj.track_id,
                "last_seen": obj.last_seen_dt,
                "has_thumbnail": obj.thumbnail_saved,
            })

    def _notify_event_update_by_id(self, tracking_id: str, status: str, ai_result: dict = None):
        if self.on_event_update:
            msg = {"tracking_id": tracking_id, "status": status}
            if ai_result:
                msg["common_name"] = ai_result.get("common_name")
                msg["scientific_name"] = ai_result.get("scientific_name")
                msg["description"] = ai_result.get("description")
            self.on_event_update(msg)

    def get_active_tracks(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "tracking_id": obj.tracking_id,
                    "track_id": obj.track_id,
                    "bbox": obj.bbox,
                    "duration": obj.last_seen - obj.first_seen,
                    "event_started": obj.event_started,
                    "ai_sent": obj.ai_sent,
                    "ended": obj.ended,
                    "last_seen": obj.last_seen_dt,
                    "has_thumbnail": obj.thumbnail_saved,
                }
                for obj in self._tracks.values()
                if not obj.ended
            ]
