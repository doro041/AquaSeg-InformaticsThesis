from ultralytics import YOLO

model = YOLO("yolo12n.pt")


model.train(data="dataset.yaml", epochs=100, imgsz=640)
