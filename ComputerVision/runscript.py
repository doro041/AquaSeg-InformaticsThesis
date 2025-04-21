from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo12n.pt")


model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    workers=4,
    save=True,         
    save_period=1      
)
