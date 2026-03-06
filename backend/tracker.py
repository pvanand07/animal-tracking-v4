"""YOLO Tracker: reads video, runs object tracking, annotates frames."""
import re
import time
import logging
import threading
from typing import Optional
import cv2
import numpy as np
from ultralytics import YOLO
from config import config
from event_manager import EventManager
from ring_buffer import RingBuffer

log = logging.getLogger("tracker")


def _engine_imgsz_from_path(path: str) -> int:
    """Infer imgsz from engine filename, e.g. yolo26n-480.engine -> 480; else 640."""
    if not path.lower().endswith(".engine"):
        return config.inference_imgsz
    match = re.search(r"-(\d+)\.engine$", path, re.IGNORECASE)
    return int(match.group(1)) if match else 640


class Tracker:
    """
    Reads video.mp4 or webcam, runs YOLO BoTSORT tracking, and:
    - Puts annotated JPEG frames into a shared buffer for streaming
    - Feeds detections into the EventManager using real wall-clock time
    - Fans out raw BGR frames to RingBuffer and RecordingCoordinator

    Jetson load reduction (config.json or env):
    - inference_interval: run YOLO every Nth frame only (default 4); higher = less GPU
    - stream_fps: cap capture/stream rate (e.g. 8–10)
    - inference_imgsz: smaller input size (320–480); must match .engine if using TensorRT
    - inference_half: true for FP16
    - stream_jpeg_quality: lower (e.g. 60) reduces CPU for encoding
    - yolo_confidence: higher = fewer detections (e.g. 0.65–0.7)
    """

    def __init__(self, event_manager: EventManager, recording_coordinator=None):
        self.event_manager = event_manager
        self.recording_coordinator = recording_coordinator
        self.model = YOLO(config.yolo_model_path)
        self.running = False
        self._thread: threading.Thread | None = None
        self._latest_frame: bytes | None = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self.frame_count = 0
        self.fps = 0.0
        self.video_finished = False
        self._last_detections: list = []
        self._last_annotations: dict | None = None
        self.ring_buffer = RingBuffer(fps=config.stream_fps, preroll_s=config.preroll_seconds)

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
            use_webcam = config.use_webcam
            if use_webcam:
                source = config.webcam_index
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    log.error(f"Cannot open webcam index {source}")
                    time.sleep(2)
                    continue
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                if video_fps < 1:
                    video_fps = 30
                log.info(f"Webcam {source} opened (streaming at {config.stream_fps} FPS)")
            else:
                source = config.video_path
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    log.error(f"Cannot open video: {source}")
                    time.sleep(2)
                    continue
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                log.info(f"Video opened: {source} at {video_fps} FPS (streaming at {config.stream_fps} FPS)")

            target_fps = min(config.stream_fps, video_fps)
            frame_interval = 1.0 / target_fps

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if use_webcam:
                        log.warning("Webcam read failed, retrying…")
                        time.sleep(0.1)
                        continue
                    break

                loop_start = time.monotonic()

                # Fan-out raw frame to ring buffer and active recorder
                self.ring_buffer.append(frame)
                if self.recording_coordinator is not None:
                    self.recording_coordinator.push_frame(frame)

                run_inference = (
                    self.frame_count % config.inference_interval == 0
                    or self._last_annotations is None
                )

                source_label = f"CAM:{config.webcam_index}" if use_webcam else "VIDEO"

                if run_inference:
                    # TensorRT .engine models use a fixed input size; infer from filename (e.g. yolo26n-480.engine) or default 640
                    model_path = config.yolo_model_path
                    imgsz = _engine_imgsz_from_path(model_path) if model_path.lower().endswith(".engine") else config.inference_imgsz
                    # Run YOLO tracking (main GPU load)
                    results = self.model.track(
                        frame,
                        persist=True,
                        conf=config.yolo_confidence,
                        verbose=False,
                        tracker=config.tracker_config,
                        imgsz=imgsz,
                        half=config.inference_half,
                    )

                    # Extract detections
                    detections = []

                    if results and results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes = results[0].boxes
                        for i in range(len(boxes)):
                            cls_id = int(boxes.cls[i].item())
                            cls_name = self.model.names.get(cls_id, "unknown")
                            # Skip humans (person class)
                            if cls_name == "person":
                                continue

                            track_id = int(boxes.id[i].item())
                            bbox = boxes.xyxy[i].cpu().numpy().tolist()
                            conf = float(boxes.conf[i].item())

                            detections.append({
                                "track_id": track_id,
                                "bbox": bbox,
                                "class_name": cls_name,
                                "confidence": conf,
                            })

                    self._last_detections = detections
                    self._last_annotations = {"detections": detections, "source_label": source_label}

                    self.event_manager.update(
                        detections, frame, loop_start, self.frame_count, skip_crop_update=False
                    )
                else:
                    # No YOLO run — reuse cached annotation data (reduces GPU load)
                    self.event_manager.update(
                        self._last_detections, frame, loop_start, self.frame_count, skip_crop_update=True
                    )

                # Stream every frame; only draw bounding boxes on inference frames
                if run_inference:
                    annotations = self._last_annotations or {"detections": [], "source_label": source_label}
                    annotated = self._draw_annotations(frame, annotations)
                else:
                    annotated = self._draw_status_overlay(frame, source_label)
                _, jpeg = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, config.stream_jpeg_quality]
                )
                frame_bytes = jpeg.tobytes()

                with self._frame_lock:
                    self._latest_frame = frame_bytes
                self._frame_event.set()
                self._frame_event.clear()

                self.frame_count += 1

                # Frame rate control — sleep first, then measure total frame duration for accurate FPS
                elapsed = time.monotonic() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                total_elapsed = time.monotonic() - loop_start
                self.fps = 1.0 / total_elapsed if total_elapsed > 0 else 0

            cap.release()

            if use_webcam:
                # Webcam disconnected — retry after a short delay
                if self.running:
                    log.warning("Webcam stream ended, retrying in 2s…")
                    time.sleep(2)
                continue

            if not config.loop_video:
                self.video_finished = True
                log.info("Video finished")
                break
            else:
                log.info("Looping video...")
                self.model = YOLO(config.yolo_model_path)

    def _draw_annotations(self, frame: np.ndarray, annotations: dict) -> np.ndarray:
        """Draw bounding boxes and overlay text onto a raw frame (inference frames only)."""
        annotated = frame.copy()
        detections = annotations.get("detections", [])
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            color = self._track_color(det["track_id"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        return annotated

    def _draw_status_overlay(self, frame: np.ndarray, source_label: str) -> np.ndarray:
        """Return frame unchanged (no top text overlay)."""
        return frame.copy()

    @staticmethod
    def _track_color(track_id: int) -> tuple:
        hue = (track_id * 47) % 180
        color_hsv = np.array([[[hue, 255, 220]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in color_bgr)

