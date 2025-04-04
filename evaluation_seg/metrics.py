import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_class_masks(image_path):
    """
    Extract binary masks for different classes based on color, whereas we get the most probable red and blue colour. Red - General/undefined lobster; Blue - Egg-berried lobsters.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None, None

    lower_blue = np.array([200, 0, 0], dtype=np.uint8)
    upper_blue = np.array([255, 50, 50], dtype=np.uint8)
  
    lower_red = np.array([0, 0, 200], dtype=np.uint8)
    upper_red = np.array([50, 50, 255], dtype=np.uint8)

    blue_mask = cv2.inRange(img, lower_blue, upper_blue)
    red_mask = cv2.inRange(img, lower_red, upper_red)
    
    return blue_mask, red_mask

def compute_iou_dice(gt_mask, pred_mask):
    """
    Compute IoU and Dice scores for the class masks.
    """
    gt_binary = (gt_mask > 0).astype(np.uint8)
    pred_binary = (pred_mask > 0).astype(np.uint8)

    if gt_binary.shape != pred_binary.shape:
        print(f"Warning: Resizing prediction mask from {pred_binary.shape} to {gt_binary.shape}")
        pred_binary = cv2.resize(pred_binary, (gt_binary.shape[1], gt_binary.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)

    # If empty masks we have edge case for division of 0
    if gt_binary.sum() == 0 and pred_binary.sum() == 0:
        return 1.0, 1.0 
    
    intersection = np.logical_and(gt_binary, pred_binary).sum()
    union = np.logical_or(gt_binary, pred_binary).sum()
    
    # Add small epsilon to prevent division by zero
    epsilon = 1e-6
    iou = intersection / (union + epsilon)
    dice = (2 * intersection) / (gt_binary.sum() + pred_binary.sum() + epsilon)

    return iou, dice

def find_matching_file(image_id, file_list, prefixes=None):
    """
    Find a matching file for the given image_id.
    """
    if prefixes:
        for prefix in prefixes:
            for file_path in file_list:
                filename = os.path.basename(file_path)
                if filename == f"{prefix}{image_id}":
                    return file_path

    for file_path in file_list:
        filename = os.path.basename(file_path)
        if image_id in filename:
            return file_path
    
    return None

def extract_image_id(filename, prefixes=None):
    """
    Extract image ID from filename.
    """
    base_name = os.path.splitext(filename)[0]

    if prefixes:
        for prefix in prefixes:
            if base_name.startswith(prefix):
                return base_name[len(prefix):]
    
    
    return base_name

def visualize_masks(gt_file, pred_file, output_dir):
    """
    Save visualization of ground truth vs prediction masks for visual inspection.
    """
    img_gt = cv2.imread(gt_file)
    img_pred = cv2.imread(pred_file)
    
    if img_gt is None or img_pred is None:
        return
    
    # Edge case for when the image shape of the ground truth is different than the predicted image 
    if img_gt.shape != img_pred.shape:
        img_pred = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]))
    
    #Compare them 
    comparison = np.hstack((img_gt, img_pred))
    

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Ground Truth", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Prediction", (img_gt.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
    # save the visualisation
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(gt_file)
    output_path = os.path.join(output_dir, f"comparison_{filename}")
    cv2.imwrite(output_path, comparison)

def evaluate_masks(gt_folder, pred_folder, output_dir=None, gt_prefix="visualized_", pred_prefix="segmented_"):
    """
    Evaluate IoU and Dice scores for both classes with improved matching and visualization.
    """
    gt_prefixes = [gt_prefix, ""] 
    pred_prefixes = [pred_prefix, ""] 
    
    # Find all image files( we have to use .png and jpg, but this function adds various format for future work expansion)
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    gt_files = []
    for ext in extensions:
        gt_files.extend(glob.glob(os.path.join(gt_folder, f"*{ext}")))
    
    pred_files = []
    for ext in extensions:
        pred_files.extend(glob.glob(os.path.join(pred_folder, f"*{ext}")))
    
    print(f"Found {len(gt_files)} ground truth files and {len(pred_files)} prediction files")
    
    iou_scores, dice_scores = {"blue": [], "red": []}, {"blue": [], "red": []}
    matched_count = 0
    
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Progress bar
    for gt_file in tqdm(gt_files, desc="Evaluating masks"):
        gt_filename = os.path.basename(gt_file)
        
      
        image_id = extract_image_id(gt_filename, gt_prefixes)
        
   
        pred_file = find_matching_file(image_id, pred_files, pred_prefixes)
        
        if pred_file:
            print(f"Matching: {gt_filename} with {os.path.basename(pred_file)}")
            matched_count += 1
            
            # Extract the masks 
            gt_blue, gt_red = extract_class_masks(gt_file)
            pred_blue, pred_red = extract_class_masks(pred_file)
            
            if gt_blue is not None and pred_blue is not None:
                # Compute IoU and Dice scores for the masks
                iou_b, dice_b = compute_iou_dice(gt_blue, pred_blue)
                iou_r, dice_r = compute_iou_dice(gt_red, pred_red)
                
                iou_scores["blue"].append(iou_b)
                dice_scores["blue"].append(dice_b)
                iou_scores["red"].append(iou_r)
                dice_scores["red"].append(dice_r)
                
                print(f"{image_id} - IoU (Blue): {iou_b:.4f}, Dice (Blue): {dice_b:.4f} | IoU (Red): {iou_r:.4f}, Dice (Red): {dice_r:.4f}")
                
                
                if output_dir:
                    visualize_masks(gt_file, pred_file, output_dir)
        else:
            print(f"Warning: No prediction found for {gt_filename} (ID: {image_id})")
    
    print(f"\nSuccessfully matched {matched_count} file pairs out of {len(gt_files)} ground truth files")
    
    # Compute the metrics
    if not iou_scores["blue"]:
        print("No valid pairs of ground truth and prediction masks were found.")
        return
    

    mean_iou_blue = np.mean(iou_scores['blue'])
    mean_dice_blue = np.mean(dice_scores['blue'])
    mean_iou_red = np.mean(iou_scores['red'])
    mean_dice_red = np.mean(dice_scores['red'])
    
    print("\nFinal Evaluation:")
    print(f"Mean IoU (Blue): {mean_iou_blue:.4f}, Mean Dice (Blue): {mean_dice_blue:.4f}")
    print(f"Mean IoU (Red): {mean_iou_red:.4f}, Mean Dice (Red): {mean_dice_red:.4f}")
    

    if output_dir:
        generate_metrics_visualization(iou_scores, dice_scores, output_dir)
    
    return {
        "iou_blue": mean_iou_blue,
        "dice_blue": mean_dice_blue,
        "iou_red": mean_iou_red,
        "dice_red": mean_dice_red,
        "individual_scores": {
            "iou_blue": iou_scores["blue"],
            "dice_blue": dice_scores["blue"],
            "iou_red": iou_scores["red"],
            "dice_red": dice_scores["red"]
        }
    }

def generate_metrics_visualization(iou_scores, dice_scores, output_dir):
    """
    Generate and save visualizations of evaluation metrics.
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot IoU scores and Dice scores for the Blue(egg-berried lobsters) 
    axs[0, 0].hist(iou_scores["blue"], bins=20, color='blue', alpha=0.7)
    axs[0, 0].set_title(f"IoU Scores - Blue Class (Mean: {np.mean(iou_scores['blue']):.4f})")
    axs[0, 0].set_xlabel("IoU Score")
    axs[0, 0].set_ylabel("Frequency")

    axs[0, 1].hist(dice_scores["blue"], bins=20, color='blue', alpha=0.7)
    axs[0, 1].set_title(f"Dice Scores - Blue Class (Mean: {np.mean(dice_scores['blue']):.4f})")
    axs[0, 1].set_xlabel("Dice Score")
    axs[0, 1].set_ylabel("Frequency")
    
      # Plot IoU scores and Dice scores for the Blue(generic lobsters) 
    axs[1, 0].hist(iou_scores["red"], bins=20, color='red', alpha=0.7)
    axs[1, 0].set_title(f"IoU Scores - Red Class (Mean: {np.mean(iou_scores['red']):.4f})")
    axs[1, 0].set_xlabel("IoU Score")
    axs[1, 0].set_ylabel("Frequency")
    

    axs[1, 1].hist(dice_scores["red"], bins=20, color='red', alpha=0.7)
    axs[1, 1].set_title(f"Dice Scores - Red Class (Mean: {np.mean(dice_scores['red']):.4f})")
    axs[1, 1].set_xlabel("Dice Score")
    axs[1, 1].set_ylabel("Frequency")
    
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_metrics.png"))
    plt.close()

if __name__ == "__main__":
    gt_path = r"C:\Users\dorot\Desktop\Dissertation2025\ModelDevelopment\Pipeline\evaluation_seg\visualized_masks_v2"
    pred_path = r"C:\Users\dorot\Desktop\Dissertation2025\ModelDevelopment\Pipeline\inference\runs\sam2s"
    output_path = r"C:\Users\dorot\Desktop\Dissertation2025\ModelDevelopment\Pipeline\evaluation_results\sam2V2"
    
    evaluate_masks(gt_path, pred_path, output_path)