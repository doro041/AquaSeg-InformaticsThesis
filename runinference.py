from ultralytics import YOLO

model = YOLO('yolo11n_lobster.pt')

results = model.predict(
    source='egg-bearing-lobsters7.png', 
    conf=0.3, 
    save=True,  
    show=True,  
    show_labels=True,
    show_conf=True
)
