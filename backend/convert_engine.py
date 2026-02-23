from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("best.pt")

# Export the model to TensorRT
model.export(format="engine")  # creates 'yolo26n.engine'
