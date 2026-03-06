from ultralytics import YOLO
model = YOLO("best.pt")
# Export at 480 for faster inference; use half=True for FP16 (default on Jetson)
model.export(format="engine", imgsz=480, half=True)