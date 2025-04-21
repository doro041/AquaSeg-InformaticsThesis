from ultralytics import YOLO

model = YOLO("yolo11n-egg_noeggseg.pt")


model.export(format="tflite") 

