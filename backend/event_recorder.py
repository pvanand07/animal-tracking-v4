"""
GStreamer appsrc-based event recorder.

Accepts preroll frames (from RingBuffer snapshot) followed by live frames
and encodes them into a single MP4 file tagged with vid_id.
"""
import configparser
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("event_recorder")

# Sentinel pushed into the frame queue to signal EOS.
_STOP = object()


class EventRecorder:
    """
    GStreamer appsrc pipeline that encodes preroll + live BGR frames to MP4.

    Usage:
        rec = EventRecorder(vid_id, output_path, width, height, fps)
        rec.start(preroll_frames)       # list[(monotonic_ts, frame)]
        rec.push_frame(frame)           # called for each live frame
        rec.stop()                      # sends EOS and blocks until done
    """

    def __init__(
        self,
        vid_id: str,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        config_ini_path: str = "config.ini",
    ):
        self.vid_id = vid_id
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.config_ini_path = config_ini_path

        self._frame_queue: queue.Queue = queue.Queue(maxsize=256)
        self._thread: Optional[threading.Thread] = None
        self._started_at: Optional[float] = None   # wall clock
        self._ended_at: Optional[float] = None
        self._lock = threading.Lock()
        self._done_event = threading.Event()
        self.failed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, preroll_frames: list[tuple[float, np.ndarray]]) -> None:
        """
        Start the encoding thread and pre-populate the queue with preroll frames.
        preroll_frames: ordered list of (monotonic_ts, BGR_frame).
        """
        self._started_at = time.time()
        preroll_queued = 0
        for item in preroll_frames:
            try:
                self._frame_queue.put_nowait(item)
                preroll_queued += 1
            except queue.Full:
                log.warning("[%s] Preroll frame dropped — queue full", self.vid_id[:8])
        log.info(
            "EventRecorder starting: vid_id=%s output=%s preroll_frames=%d/%d",
            self.vid_id, self.output_path, preroll_queued, len(preroll_frames),
        )
        self._thread = threading.Thread(
            target=self._encode_loop,
            name=f"EventRecorder-{self.vid_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        log.info("EventRecorder thread launched: %s", self.vid_id)

    def push_frame(self, frame: np.ndarray) -> bool:
        """
        Queue a live BGR frame. Returns False and drops the frame if the
        queue is full (backpressure protection).
        """
        ts = time.monotonic()
        try:
            self._frame_queue.put_nowait((ts, frame))
            return True
        except queue.Full:
            return False

    def stop(self) -> None:
        """Signal EOS and block until the pipeline finishes."""
        self._frame_queue.put(_STOP)
        self._done_event.wait(timeout=30)
        self._ended_at = time.time()
        log.info("EventRecorder stopped: %s (duration %.1fs)", self.vid_id, self.duration_s or 0)

    @property
    def duration_s(self) -> Optional[float]:
        if self._started_at is None:
            return None
        end = self._ended_at or time.time()
        return end - self._started_at

    # ------------------------------------------------------------------
    # Internal encoding thread
    # ------------------------------------------------------------------

    def _encode_loop(self) -> None:
        log.info("[%s] Encode loop starting", self.vid_id[:8])
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst, GLib
        except Exception as exc:
            log.error("[%s] GStreamer unavailable: %s", self.vid_id[:8], exc)
            self.failed = True
            self._done_event.set()
            return

        log.debug("[%s] GStreamer imported OK", self.vid_id[:8])
        Gst.init(None)

        # Read overlay / encoding settings from config.ini
        cfg = configparser.ConfigParser()
        if os.path.exists(self.config_ini_path):
            cfg.read(self.config_ini_path)
            log.debug("[%s] Loaded config.ini from %s", self.vid_id[:8], self.config_ini_path)
        else:
            log.warning("[%s] config.ini not found at %s — using defaults", self.vid_id[:8], self.config_ini_path)

        e_sec = cfg["encoding"] if cfg.has_section("encoding") else {}
        o_sec = cfg["overlay"] if cfg.has_section("overlay") else {}

        bitrate = e_sec.get("bitrate", "4000")
        speed_preset = e_sec.get("speed_preset", "ultrafast")
        overlay_text = o_sec.get("text", "AC Future")
        halign = o_sec.get("halign", "right")
        valign = o_sec.get("valign", "top")
        font_size = o_sec.get("font_size", "20")
        text_ypad = o_sec.get("text_ypad", "10")
        temp_ypad = o_sec.get("temp_ypad", "45")
        clock_ypad = o_sec.get("clock_ypad", "80")

        fps_num = max(1, int(round(self.fps)))
        log.info(
            "[%s] Encoding params: %dx%d @ %dfps, bitrate=%s, output=%s",
            self.vid_id[:8], self.width, self.height, fps_num, bitrate, self.output_path,
        )

        # Ensure output directory exists before the pipeline tries to create the file
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        pipeline_str = (
            f"appsrc name=src is-live=false format=time "
            f"caps=video/x-raw,format=BGR,"
            f"width={self.width},height={self.height},"
            f"framerate={fps_num}/1 ! "
            f"videoconvert ! video/x-raw,format=I420 ! "
            f"textoverlay name=static_txt "
            f"text=\"{overlay_text}\" "
            f"halignment={halign} valignment={valign} "
            f"ypad={text_ypad} font-desc=\"Sans {font_size}\" ! "
            f"textoverlay name=temp_txt "
            f"halignment={halign} valignment={valign} "
            f"ypad={temp_ypad} font-desc=\"Sans {font_size}\" ! "
            f"clockoverlay "
            f"halignment={halign} valignment={valign} "
            f"ypad={clock_ypad} "
            f"time-format=\"%Y-%m-%d %H:%M:%S\" "
            f"font-desc=\"Sans {font_size}\" ! "
            f"x264enc tune=zerolatency "
            f"speed-preset={speed_preset} bitrate={bitrate} ! "
            f"h264parse ! qtmux ! "
            f"filesink location={self.output_path}"
        )

        log.info("[%s] Pipeline: %s", self.vid_id[:8], pipeline_str)

        try:
            pipeline = Gst.parse_launch(pipeline_str)
        except Exception as exc:
            log.error("[%s] Gst.parse_launch failed: %s", self.vid_id[:8], exc)
            self.failed = True
            self._done_event.set()
            return

        if pipeline is None:
            log.error("[%s] Gst.parse_launch returned None — pipeline string invalid", self.vid_id[:8])
            self.failed = True
            self._done_event.set()
            return

        appsrc = pipeline.get_by_name("src")
        temp_elem = pipeline.get_by_name("temp_txt")

        if appsrc is None:
            log.error("[%s] appsrc element 'src' not found in pipeline", self.vid_id[:8])
            self.failed = True
            self._done_event.set()
            return

        log.debug("[%s] Pipeline elements OK (appsrc=%s, temp_txt=%s)", self.vid_id[:8], appsrc, temp_elem)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        loop = GLib.MainLoop()

        eos_received = threading.Event()

        def on_message(bus, msg):
            t = msg.type
            if t == Gst.MessageType.EOS:
                log.info("[%s] GStreamer EOS received", self.vid_id[:8])
                eos_received.set()
                loop.quit()
            elif t == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                log.error("[%s] GStreamer ERROR: %s | debug: %s", self.vid_id[:8], err.message, dbg)
                self.failed = True
                eos_received.set()
                loop.quit()
            elif t == Gst.MessageType.WARNING:
                warn, dbg = msg.parse_warning()
                log.warning("[%s] GStreamer WARNING: %s | debug: %s", self.vid_id[:8], warn.message, dbg)
            elif t == Gst.MessageType.STATE_CHANGED:
                if msg.src == pipeline:
                    old, new, pending = msg.parse_state_changed()
                    log.debug(
                        "[%s] Pipeline state: %s → %s (pending: %s)",
                        self.vid_id[:8],
                        Gst.Element.state_get_name(old),
                        Gst.Element.state_get_name(new),
                        Gst.Element.state_get_name(pending),
                    )

        bus.connect("message", on_message)

        def _update_temp():
            if temp_elem:
                try:
                    with open("/sys/class/thermal/thermal_zone0/temp") as f:
                        temp = int(f.read()) / 1000.0
                    temp_elem.set_property("text", f"CPU: {temp:.1f}C")
                except Exception:
                    pass
            return True  # keep timer alive

        # Start the GLib main loop in a sub-thread (handles bus messages)
        glib_thread = threading.Thread(target=loop.run, daemon=True, name=f"GLib-{self.vid_id[:8]}")
        glib_thread.start()

        # Transition to PLAYING and wait for confirmation
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            log.error("[%s] Pipeline failed to reach PLAYING state", self.vid_id[:8])
            self.failed = True
            loop.quit()
            self._done_event.set()
            return
        log.info("[%s] Pipeline set_state(PLAYING) → %s", self.vid_id[:8], ret.value_name)

        GLib.timeout_add_seconds(1, _update_temp)

        frame_duration_ns = int(Gst.SECOND / self.fps)
        pts = 0
        frames_pushed = 0
        frames_dropped = 0

        while True:
            try:
                item = self._frame_queue.get(timeout=2.0)
            except queue.Empty:
                log.debug("[%s] Frame queue empty (pushed=%d so far)", self.vid_id[:8], frames_pushed)
                continue

            if item is _STOP:
                log.info("[%s] STOP sentinel received — pushed %d frames, dropped %d", self.vid_id[:8], frames_pushed, frames_dropped)
                break

            _ts, frame = item

            # Ensure frame dimensions match
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                import cv2
                frame = cv2.resize(frame, (self.width, self.height))
                log.debug("[%s] Resized frame to %dx%d", self.vid_id[:8], self.width, self.height)

            buf = Gst.Buffer.new_wrapped(frame.tobytes())
            buf.pts = pts
            buf.duration = frame_duration_ns
            pts += frame_duration_ns

            ret = appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                log.error("[%s] appsrc push-buffer returned %s after %d frames", self.vid_id[:8], ret, frames_pushed)
                frames_dropped += 1
                break
            else:
                frames_pushed += 1
                if frames_pushed % 30 == 0:
                    log.debug("[%s] Pushed %d frames (%.1fs encoded)", self.vid_id[:8], frames_pushed, frames_pushed / self.fps)

        log.info("[%s] Sending EOS to pipeline (total pushed=%d)", self.vid_id[:8], frames_pushed)
        appsrc.emit("end-of-stream")

        if not eos_received.wait(timeout=15):
            log.error("[%s] Timed out waiting for EOS after 15s — pipeline may be hung", self.vid_id[:8])
        else:
            log.info("[%s] EOS confirmed, setting pipeline to NULL", self.vid_id[:8])

        pipeline.set_state(Gst.State.NULL)
        loop.quit()

        # Verify output file was actually written
        out = Path(self.output_path)
        if out.exists():
            size_kb = out.stat().st_size // 1024
            log.info("[%s] Recording finalised: %s (%d KB, %d frames)", self.vid_id[:8], self.output_path, size_kb, frames_pushed)
        else:
            log.error("[%s] Output file NOT found after pipeline: %s", self.vid_id[:8], self.output_path)

        self._done_event.set()
