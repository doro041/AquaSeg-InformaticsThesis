from ultralytics import FastSAM

# Load the YOLO11 model
model = FastSAM("FastSAM-s.pt")

model.export(format="tflite")

