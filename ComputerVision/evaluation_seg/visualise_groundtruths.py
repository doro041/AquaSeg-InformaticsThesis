import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Paths
image_folder = r"C:\Users\dorot\Desktop\Dissertation2025\ModelDevelopment\Pipeline\evaluation_seg\segGT\segGT\images"
label_folder = r"C:\Users\dorot\Desktop\Dissertation2025\ModelDevelopment\Pipeline\evaluation_seg\segGT\segGT\labels"
output_folder = r"C:\Users\dorot\Desktop\Dissertation2025\ModelDevelopment\Pipeline\evaluation_seg\visualized_masks_v2"

os.makedirs(output_folder, exist_ok=True)

colors = [
    (0, 0, 1, 0.5), # Blue - egg-berried  lobster
    (1, 0, 0, 0.5)   # Red - general/undefined lobster
]
'''Something that needs to be noted is the following: the format of the labels in the folder are in the format of YOLO-11/YOLO-12 for image segmentation, which means class_id, polygons. '''
# Process files
for image_file in os.listdir(image_folder):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
        # Get the label
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + ".txt"
        label_path = os.path.join(label_folder, label_file)
        

        if not os.path.exists(label_path):
            print(f"No label file for {image_file}. Skipping...")
            continue
            
        # Read image
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image {image_file}. Skipping...")
            continue
            
        #from BGR to RGB 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions for normalization
        img_height, img_width = img.shape[:2]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
       
        polygons = []
        polygon_colors = []
      
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:  
                    continue
                    
                class_id = int(parts[0])
                # we find the polygon coordinates
                polygon_points = []
                for i in range(1, len(parts), 2):
                    if i+1 < len(parts):
                        x = float(parts[i]) * img_width
                        y = float(parts[i+1]) * img_height
                        polygon_points.append((x, y))
                
                if len(polygon_points) > 2: 
                    polygons.append(Polygon(np.array(polygon_points)))
                    polygon_colors.append(colors[class_id % len(colors)])
        
        # add the polygon to the image
        p = PatchCollection(polygons, facecolors=polygon_colors, edgecolors=(0,0,0,1), linewidths=2)
        ax.add_collection(p)
        
        # adding title of image
        ax.set_title(f'Image: {image_file}')
        ax.axis('off')

        output_path = os.path.join(output_folder, f"visualized_{base_name}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Saved visualization for {image_file}")

print("Visualization complete!")