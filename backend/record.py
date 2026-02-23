import gi
import configparser
import datetime
import os
import sys

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class VideoRecorder:
    def __init__(self, config_path='config.ini'):
        Gst.init(None)
        
        # Load Config
        config = configparser.ConfigParser()
        if not os.path.exists(config_path):
            print(f"Error: {config_path} not found.")
            sys.exit(1)
        config.read(config_path)
        
        c = config['camera']
        e = config['encoding']
        o = config['overlay']
        
        self.duration = int(c.get('duration', 0))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"capture_{timestamp}.mp4"

        # Pipeline: use jpegdec + videoconvert (nvjpegdec often fails on Jetson due to libjpeg symbol mismatch)
        pipeline_str = (
            f"v4l2src device={c['device']} ! "
            f"image/jpeg, width={c['width']}, height={c['height']}, framerate={c['framerate']}/1 ! "
            f"jpegdec ! videoconvert ! video/x-raw, format=I420 ! "
            f"textoverlay name=static_txt text=\"{o['text']}\" halignment={o['halign']} valignment={o['valign']} "
            f"ypad={o['text_ypad']} font-desc=\"Sans {o['font_size']}\" ! "
            f"textoverlay name=temp_txt halignment={o['halign']} valignment={o['valign']} "
            f"ypad={o['temp_ypad']} font-desc=\"Sans {o['font_size']}\" ! "
            f"clockoverlay halignment={o['halign']} valignment={o['valign']} "
            f"ypad={o['clock_ypad']} time-format=\"%Y-%m-%d %H:%M:%S\" font-desc=\"Sans {o['font_size']}\" ! "
            f"x264enc tune=zerolatency speed-preset={e['speed_preset']} bitrate={e['bitrate']} ! "
            f"h264parse ! qtmux ! filesink location={output_file}"
        )

        print(f"--- Starting Recording to {output_file} ---")
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.temp_element = self.pipeline.get_by_name("temp_txt")
        self.loop = GLib.MainLoop()
        
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

    def get_cpu_temp(self):
        try:
            # Standard path for Jetson Orin Nano thermal data
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = int(f.read()) / 1000.0
                return f"CPU: {temp:.1f}C"
        except Exception:
            return "CPU: --C"

    def update_temp_overlay(self):
        if self.temp_element:
            self.temp_element.set_property("text", self.get_cpu_temp())
        return True 

    def on_message(self, bus, message):
        if message.type == Gst.MessageType.EOS:
            self.loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"GStreamer Error: {err.message}")
            self.stop()

    def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # Update temp every second
        GLib.timeout_add_seconds(1, self.update_temp_overlay)

        if self.duration > 0:
            print(f"Auto-stop in {self.duration}s. Monitoring CPU...")
            GLib.timeout_add_seconds(self.duration, self.timeout_callback)
        else:
            print("Recording... (Ctrl+C to stop)")

        try:
            self.loop.run()
        except KeyboardInterrupt:
            self.stop()

    def timeout_callback(self):
        print("\nTimer finished.")
        self.stop()
        return False

    def stop(self):
        self.pipeline.send_event(Gst.Event.new_eos())
        self.pipeline.get_bus().timed_pop_filtered(2 * Gst.SECOND, Gst.MessageType.EOS)
        self.pipeline.set_state(Gst.State.NULL)
        print("Recording saved successfully.")
        if self.loop.is_running():
            self.loop.quit()

if __name__ == "__main__":
    recorder = VideoRecorder()
    recorder.start()
