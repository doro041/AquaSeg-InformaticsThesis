from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo12n.pt")


model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    workers=4 
)
