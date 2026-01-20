import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim, mean_squared_error
import matplotlib.pyplot as plt

def read_and_preprocess_image(file_path, invert=False):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if invert:
        img = cv2.bitwise_not(img)
    return img

def crop_bbox(image, bbox):
    x = int(bbox['bbox_x'])
    y = int(bbox['bbox_y'])
    w = int(bbox['bbox_width'])
    h = int(bbox['bbox_height'])
    h_img, w_img = image.shape
    x = max(0, min(x, w_img))
    y = max(0, min(y, h_img))
    w = max(0, min(w, w_img - x))
    h = max(0, min(h, h_img - y))
    return image[y:y+h, x:x+w]

def resize_to_match(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    return img2

def calculate_object_metrics(gt_crop, seg_crop, label):
    seg_crop = resize_to_match(gt_crop, seg_crop)
    mse = mean_squared_error(gt_crop.astype(float) / 255.0, seg_crop.astype(float) / 255.0)
    if gt_crop.shape[0] >= 7 and gt_crop.shape[1] >= 7:
        ssim_value = ssim(gt_crop, seg_crop, data_range=255)
    else:
        ssim_value = None
    return {
        'label': label,
        'mse': mse,
        'ssim': ssim_value
    }

def match_boxes(gt_df, seg_df):
    def calculate_iou(box1, box2):
        x1_min = box1['bbox_x']
        y1_min = box1['bbox_y']
        x1_max = box1['bbox_x'] + box1['bbox_width']
        y1_max = box1['bbox_y'] + box1['bbox_height']
        
        x2_min = box2['bbox_x']
        y2_min = box2['bbox_y']
        x2_max = box2['bbox_x'] + box2['bbox_width']
        y2_max = box2['bbox_y'] + box2['bbox_height']
        
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = box1['bbox_width'] * box1['bbox_height']
        area2 = box2['bbox_width'] * box2['bbox_height']
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    matches = []
    matched_seg_indices = set()
    
    for gt_idx, gt_box in gt_df.iterrows():
        best_iou = 0
        best_seg_idx = None
        
        for seg_idx, seg_box in seg_df.iterrows():
            if seg_idx in matched_seg_indices:
                continue
            iou = calculate_iou(gt_box, seg_box)
            if iou > best_iou:
                best_iou = iou
                best_seg_idx = seg_idx
        
        if best_seg_idx is not None:
            matched_seg_indices.add(best_seg_idx)
            matches.append((gt_idx, best_seg_idx))
    return matches

if __name__ == "__main__":
    gt_image_path = 'new_output/gnd_truth_new_plain.jpeg'
    gt_bbox_csv = 'new_output/gnd_truth_plain_bbox.csv'
    seg_image_path = 'new_output/SAM_output.jpeg'
    seg_bbox_csv = 'new_output/sam_iamge_bbox.csv'
    # seg_image_path = 'new_output/opencv_segmentation.jpeg'
    # seg_bbox_csv = 'new_output/segmented_image_bbox.csv'
    
    print("OBJECT-LEVEL SSIM AND MSE CALCULATION")
    print("\nLoading images...")
    gt_image = read_and_preprocess_image(gt_image_path, invert=False)
    seg_image = read_and_preprocess_image(seg_image_path, invert=True)
    
    print("Loading bounding boxes...")
    gt_df = pd.read_csv(gt_bbox_csv)
    seg_df = pd.read_csv(seg_bbox_csv)
    
    matches = match_boxes(gt_df, seg_df)
    results = []
    print("\n")
    print("PER-OBJECT METRICS \n")
    
    for gt_idx, seg_idx in matches:
        gt_box = gt_df.iloc[gt_idx]
        seg_box = seg_df.iloc[seg_idx]
        
        gt_crop = crop_bbox(gt_image, gt_box)
        seg_crop = crop_bbox(seg_image, seg_box)
        
        metrics = calculate_object_metrics(gt_crop, seg_crop, gt_box['label_name'])
        results.append(metrics)
        
        print(f"\nObject {len(results)}: {metrics['label']}")
        print(f"MSE: {metrics['mse']:.6f}")
        if metrics['ssim'] is not None:
            print(f"SSIM: {metrics['ssim']:.4f}")
        else:
            print(f"SSIM: N/A (image too small)")
    
    print("\n")
    print("OVERALL METRICS (Full Image) \n")
    
    if gt_image.shape!= seg_image.shape:
        seg_image_resized = cv2.resize(seg_image,(gt_image.shape[1], gt_image.shape[0]))
    else:
        seg_image_resized = seg_image

    overall_mse = mean_squared_error(
        gt_image.astype(float) / 255.0,
        seg_image_resized.astype(float) / 255.0
    )
    overall_ssim = ssim(gt_image, seg_image_resized, data_range=255)
    
    print(f"Full Image MSE: {overall_mse:.6f}")
    print(f"Full Image SSIM: {overall_ssim:.4f}")
    
    valid_ssims = [r['ssim'] for r in results if r['ssim'] is not None]
    avg_object_mse = np.mean([r['mse'] for r in results])
    avg_object_ssim = np.mean(valid_ssims) if valid_ssims else None
    
    print("\n")
    print("AVERAGE PER-OBJECT METRICS \n")
    print(f"Average Object MSE: {avg_object_mse:.6f}")
    if avg_object_ssim is not None:
        print(f"Average Object SSIM: {avg_object_ssim:.4f}")
    else:
        print(f"Average Object SSIM: N/A")
    
    print("\nGenerating visualization...")
    
    difference = cv2.absdiff(gt_image, seg_image_resized)
    normalized_difference = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gt_image, cmap='gray')
    axes[0].set_title('Ground Truth', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(seg_image, cmap='gray')
    axes[1].set_title('Segmented Output', fontsize=10, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(normalized_difference, cmap='hot')
    axes[2].set_title(f'SAM Difference Map\nMSE: {overall_mse:.6f} | SSIM: {overall_ssim:.4f}', 
                     fontsize=10, fontweight='bold')
    axes[2].axis('off')
    plt.tight_layout()
    # plt.savefig('overall_image_comparison.png', dpi=150, bbox_inches='tight')
    # print("Saved visualization to: overall_image_comparison.png")
    plt.show()