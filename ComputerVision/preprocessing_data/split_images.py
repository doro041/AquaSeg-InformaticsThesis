import os
import shutil
import random

# set paths
dataset_path = "lobster_data"
images_path = os.path.join(dataset_path, "images")

# defining split ratios
train_ratio = 0.8
val_ratio = 0.2
test_ratio = 0.1


split_dirs = ["train", "val", "test"]
for split in split_dirs:
    os.makedirs(os.path.join(images_path, split), exist_ok=True)

# Get all image files 
image_files = [f for f in os.listdir(images_path) if f.endswith((".jpg"))]
image_files = [os.path.join(images_path, f) for f in image_files] 

# shuffle data 
random.seed(42) 
random.shuffle(image_files)

# compute split indices
num_images = len(image_files)
train_idx = int(num_images * train_ratio)
val_idx = train_idx + int(num_images * val_ratio)

# splitting datasets
train_files = image_files[:train_idx]
val_files = image_files[train_idx:val_idx]
test_files = image_files[val_idx:]

# move images
def move_images(files, split):
    for img_src in files:
        file_name = os.path.basename(img_src)  
        img_dst = os.path.join(images_path, split, file_name)  
        shutil.move(img_src, img_dst) 


move_images(train_files, "train")
move_images(val_files, "val")
move_images(test_files, "test")

print(" Task achieved.")
