from ultralytics import YOLO

model = YOLO("yolo112n.pt")


model.train(data="dataset.yaml", epochs=100, imgsz=640)
