import os
import torch
from ultralytics import FastSAM,YOLO
import cv2
import numpy as np

# load the yolo model first
yolo_model = YOLO("models/yolo12n_egg_noegg.pt")
sam_model = FastSAM("models/FastSAM-s.pt")

input_dir = r'C:\Users\dorot\Desktop\Dissertation2025\ModelDevelopment\Pipeline\evaluation_seg\segGT\segGT\images'
output_dir = "runs/fastsam"
os.makedirs(output_dir, exist_ok=True)

# extract class name from yolo
class_names = yolo_model.model.names

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        # load image
        image_path = os.path.join(input_dir, filename)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Error loading image: {image_path}")
            continue

        try:
            yolo_results = yolo_model(image_path, conf=0.7)

            if not yolo_results[0].boxes or len(yolo_results[0].boxes.xyxy) == 0:
                print(f"No lobsters detected in {filename}")
                continue

            for i, (box, conf, cls) in enumerate(zip(yolo_results[0].boxes.xyxy, yolo_results[0].boxes.conf, yolo_results[0].boxes.cls)):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                confidence = float(conf.cpu().item())
                class_id = int(cls.cpu().item())

              
                print(f"YOLO detected class ID: {class_id} with confidence {confidence:.2f} at [{x1},{y1},{x2},{y2}]")

                #we validate the bounding boxes
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                    print(f"Invalid bounding box: [{x1},{y1},{x2},{y2}] - skipping")
                    continue

                # we add a small padding for the sam model predictions
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(img.shape[1], x2 + padding)
                y2 = min(img.shape[0], y2 + padding)

                # we run sam-variant model
                try:
                    sam_results = sam_model(image_path, bboxes=[[x1, y1, x2, y2]])

                    if not hasattr(sam_results[0], "masks") or sam_results[0].masks is None or len(sam_results[0].masks) == 0:
                        print(f"SAM did not generate masks for detection {i} in {filename}")
                        continue

                    # get the segmentation masks
                    mask = sam_results[0].masks.data[0].cpu().numpy()

                
                    overlay = np.zeros_like(img, dtype=np.uint8)

                  
                    if class_id == 1:  # Egg-bearing lobster
                        overlay[mask > 0.5] = [0, 0, 255]  # Blue for egg-bearing lobsters
                    else:  # Undefined 
                        overlay[mask > 0.5] = [255, 0, 0]  # Red for the generic lobster

                    
                    alpha = 0.5  
                    blended_img = cv2.addWeighted(img, 1, overlay, alpha, 0)

                    # save the blended image
                    mask_output_path = os.path.join(output_dir, f"segmented_{filename}")
                    cv2.imwrite(mask_output_path, blended_img)

                    print(f"Processed {filename} and saved segmentation overlay: {mask_output_path}")

                except Exception as e:
                    print(f"Error processing detection {i} in {filename} with SAM: {e}")

        except Exception as e:
            print(f"Error processing {filename} with YOLO: {e}")
            continue
