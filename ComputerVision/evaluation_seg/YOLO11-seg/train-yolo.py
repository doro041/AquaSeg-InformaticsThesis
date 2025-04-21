import os
from ultralytics import YOLO

# Path to your last checkpoint
path = 'yolov11seg_runs/experiment1/weights/last.pt'

if os.path.exists(path):
    try:
        model = YOLO(path)
        print("✅ Model loaded successfully.")

        # Train for 20 more epochs
        model.train(
            data='data.yaml',
            epochs=20,  
            imgsz=640,
            batch=16,
            patience=50,
            device='cpu',
            workers=8,
            project='yolov11seg_runs',
            name='experiment1',
            save=True,
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            val=True,
            save_period=1  
        )

    except Exception as e:
        print(f"❌ Failed to load or train: {e}")
else:
    print("❌ File does not exist.")
