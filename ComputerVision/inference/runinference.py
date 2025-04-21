import os
from ultralytics import YOLO


model = YOLO('yolo12n_egg_noegg.pt')

source_folder = r'C:\Users\dorot\Desktop\Dissertation2025\MobileDevelopment\LobsterDataset2025\Aquaseg_Lobster_Dataset\tryinf' 

output_folder = 'yolo12outputboxes'

os.makedirs(output_folder, exist_ok=True)


for image_name in os.listdir(source_folder):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(source_folder, image_name)
        
       
        results = model(
            source=image_path, 
            conf=0.9, 
            save=True,  
            save_dir=output_folder,
            show=False, 
            show_labels=True,
            show_conf=True
        )

print("Inference complete. Results saved to:", output_folder)
