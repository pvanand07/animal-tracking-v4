"""YOLO Tracker: reads video, runs object tracking, annotates frames."""
import time
import logging
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from config import config
from event_manager import EventManager

log = logging.getLogger("tracker")


class Tracker:
    """
    Reads video.mp4, runs YOLO BoTSORT tracking, and:
    - Puts annotated JPEG frames into a shared buffer for streaming
    - Feeds detections into the EventManager using real wall-clock time
    """

    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
        self.model = YOLO(config.yolo_model)
        self.running = False
        self._thread: threading.Thread | None = None
        self._latest_frame: bytes | None = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self.frame_count = 0
        self.fps = 0.0
        self.video_finished = False

    @property
    def latest_frame(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_frame

    def wait_for_frame(self, timeout: float = 1.0) -> bool:
        return self._frame_event.wait(timeout)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.running = True
        self.video_finished = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("Tracker started")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Tracker stopped")

    def _run(self):
        while self.running:
            cap = cv2.VideoCapture(config.video_path)
            if not cap.isOpened():
                log.error(f"Cannot open video: {config.video_path}")
                time.sleep(2)
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            target_fps = min(config.stream_fps, video_fps)
            frame_interval = 1.0 / target_fps

            log.info(f"Video opened: {config.video_path} at {video_fps} FPS (streaming at {target_fps} FPS)")

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                loop_start = time.monotonic()

                # Run YOLO tracking
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=config.yolo_confidence,
                    verbose=False,
                    tracker=config.tracker_config,
                )

                # Extract detections
                detections = []
                annotated = frame.copy()

                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        track_id = int(boxes.id[i].item())
                        bbox = boxes.xyxy[i].cpu().numpy().tolist()
                        conf = float(boxes.conf[i].item())
                        cls_id = int(boxes.cls[i].item())
                        cls_name = self.model.names.get(cls_id, "unknown")

                        detections.append({
                            "track_id": track_id,
                            "bbox": bbox,
                            "class_name": cls_name,
                            "confidence": conf,
                        })

                        # Annotate frame
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        color = self._track_color(track_id)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        label = f"#{track_id} {cls_name} {conf:.2f}"
                        self._draw_label(annotated, label, x1, y1 - 8, color)

                # Draw FPS overlay
                cv2.putText(
                    annotated,
                    f"FPS: {self.fps:.1f} | Tracks: {len(detections)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 200),
                    2,
                )

                # Feed detections to event manager with real monotonic time and frame number
                self.event_manager.update(detections, frame, loop_start, self.frame_count)

                # Encode annotated frame as JPEG
                _, jpeg = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                frame_bytes = jpeg.tobytes()

                with self._frame_lock:
                    self._latest_frame = frame_bytes
                self._frame_event.set()
                self._frame_event.clear()

                self.frame_count += 1

                # Frame rate control
                elapsed = time.monotonic() - loop_start
                self.fps = 1.0 / elapsed if elapsed > 0 else 0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            cap.release()

            if not config.loop_video:
                self.video_finished = True
                log.info("Video finished")
                break
            else:
                log.info("Looping video...")
                self.model = YOLO(config.yolo_model)

    @staticmethod
    def _track_color(track_id: int) -> tuple:
        hue = (track_id * 47) % 180
        color_hsv = np.array([[[hue, 255, 220]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in color_bgr)

    @staticmethod
    def _draw_label(img, text, x, y, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        y = max(y, th + 4)
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 4), color, -1)
        brightness = sum(color) / 3
        txt_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        cv2.putText(img, text, (x + 2, y), font, scale, txt_color, thickness)
