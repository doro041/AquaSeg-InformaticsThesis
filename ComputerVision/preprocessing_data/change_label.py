import os

dataset_path = "lobster_data"
annotations_path = os.path.join(dataset_path, "labels")
for split in ["train", "val", "test"]:
    split_annotations_path = os.path.join(annotations_path, split)
    
    # Get all .txt annotation files in the split directory
    annotation_files = [f for f in os.listdir(split_annotations_path) if f.lower().endswith(".txt")]
    
    print(f"Processing {split} split - found {len(annotation_files)} annotations")
    
    # Modify the labels in each annotation file
    for annotation_file in annotation_files:
        annotation_path = os.path.join(split_annotations_path, annotation_file)
        
        # Open and read the annotation file
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
        
        # Modify the label from 1 to 0
        with open(annotation_path, 'w') as file:
            for line in lines:
                parts = line.split()
                if parts and parts[0] == '1':  
                    parts[0] = '0' 
                file.write(' '.join(parts) + '\n')  

print("Label modification completed!")