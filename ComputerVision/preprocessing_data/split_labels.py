import os
import shutil


dataset_path = "lobster_data"
images_path = os.path.join(dataset_path, "images")
annotations_path = os.path.join(dataset_path, "annotations")


for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(annotations_path, split), exist_ok=True)

# process each folder
for split in ["train", "val", "test"]:
   
    split_images_path = os.path.join(images_path, split)
    image_files = [f for f in os.listdir(split_images_path) if f.lower().endswith(".jpg")]
    
    print(f"Processing {split} split - found {len(image_files)} images")
    
    # for each image
    moved_count = 0
    for img_file in image_files:
        # Get base filename without extension
        base_name = os.path.splitext(img_file)[0]
        
        # Create the annotation filename
        annotation_file = base_name + ".txt"
        
        # Source and destination paths
        source_path = os.path.join(annotations_path, annotation_file)
        destination_path = os.path.join(annotations_path, split, annotation_file)
        
        # Check if annotation exists
        if os.path.exists(source_path):
            # Move the annotation to the correct split folder
            try:
                shutil.move(source_path, destination_path)
                moved_count += 1
            except Exception as e:
                print(f"Error moving {annotation_file}: {str(e)}")
        else:
            print(f"Warning: No annotation found for {img_file} in {split} split")
    
    print(f"Moved {moved_count} annotation files to {split} folder\n")

print(" Annotation moving completed!")