import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    box1_x_min = box1['bbox_x']
    box1_y_min = box1['bbox_y']
    box1_x_max = box1['bbox_x'] + box1['bbox_width']
    box1_y_max = box1['bbox_y'] + box1['bbox_height']
    
    box2_x_min = box2['bbox_x']
    box2_y_min = box2['bbox_y']
    box2_x_max = box2['bbox_x'] + box2['bbox_width']
    box2_y_max = box2['bbox_y'] + box2['bbox_height']
    
    x_left = max(box1_x_min, box2_x_min)
    y_top = max(box1_y_min, box2_y_min)
    x_right = min(box1_x_max, box2_x_max)
    y_bottom = min(box1_y_max, box2_y_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = box1['bbox_width'] * box1['bbox_height']
    box2_area = box2['bbox_width'] * box2['bbox_height']
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def match_boxes(gt_df, seg_df):
    """Match ground truth to segmented boxes"""
    results = []
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
            results.append({
                'gt_idx': gt_idx,
                'seg_idx': best_seg_idx,
                'iou': best_iou
            })
    
    return results

# Load data
gt_df = pd.read_csv('new_output/gnd_truth_plain_bbox.csv')
seg_df = pd.read_csv('new_output/sam_iamge_bbox.csv')

# Match boxes
matches = match_boxes(gt_df, seg_df)

# Create visualization
fig = plt.figure(figsize=(18, 5 * len(matches)))

for idx, match in enumerate(matches):
    gt_box = gt_df.iloc[match['gt_idx']]
    seg_box = seg_df.iloc[match['seg_idx']]
    iou = match['iou']
    
    # Create three subplots for each match
    ax1 = plt.subplot(len(matches), 3, idx*3 + 1)
    ax2 = plt.subplot(len(matches), 3, idx*3 + 2)
    ax3 = plt.subplot(len(matches), 3, idx*3 + 3)
    
    # Get image dimensions
    img_width = gt_box['image_width']
    img_height = gt_box['image_height']
    
    # Plot 1: Ground Truth
    ax1.set_xlim(0, img_width)
    ax1.set_ylim(img_height, 0)
    ax1.set_aspect('equal')
    gt_rect = patches.Rectangle(
        (gt_box['bbox_x'], gt_box['bbox_y']),
        gt_box['bbox_width'],
        gt_box['bbox_height'],
        linewidth=3,
        edgecolor='green',
        facecolor='none'
    )
    ax1.add_patch(gt_rect)
    ax1.set_title(f"Ground Truth\n{gt_box['label_name']}", fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Segmented
    ax2.set_xlim(0, img_width)
    ax2.set_ylim(img_height, 0)
    ax2.set_aspect('equal')
    seg_rect = patches.Rectangle(
        (seg_box['bbox_x'], seg_box['bbox_y']),
        seg_box['bbox_width'],
        seg_box['bbox_height'],
        linewidth=3,
        edgecolor='red',
        facecolor='none'
    )
    ax2.add_patch(seg_rect)
    ax2.set_title(f"Segmented\n{seg_box['label_name']}", fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Overlay
    ax3.set_xlim(0, img_width)
    ax3.set_ylim(img_height, 0)
    ax3.set_aspect('equal')
    
    # Draw both boxes
    gt_rect_overlay = patches.Rectangle(
        (gt_box['bbox_x'], gt_box['bbox_y']),
        gt_box['bbox_width'],
        gt_box['bbox_height'],
        linewidth=2,
        edgecolor='green',
        facecolor='green',
        alpha=0.3,
        label='Ground Truth'
    )
    seg_rect_overlay = patches.Rectangle(
        (seg_box['bbox_x'], seg_box['bbox_y']),
        seg_box['bbox_width'],
        seg_box['bbox_height'],
        linewidth=2,
        edgecolor='red',
        facecolor='red',
        alpha=0.3,
        label='Segmented'
    )
    ax3.add_patch(gt_rect_overlay)
    ax3.add_patch(seg_rect_overlay)
    
    # Color code by IoU quality
    if iou >= 0.75:
        color = 'darkgreen'
        quality = 'EXCELLENT'
    elif iou >= 0.5:
        color = 'orange'
        quality = 'GOOD'
    elif iou >= 0.3:
        color = 'darkorange'
        quality = 'FAIR'
    else:
        color = 'red'
        quality = 'POOR'
    
    ax3.set_title(f"Overlay - IoU: {iou:.4f}\n{quality}", 
                  fontsize=12, fontweight='bold', color=color)
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('iou_cvvgnd.png', dpi=150, bbox_inches='tight')
plt.savefig('iou_samvgnd.png', dpi=150, bbox_inches='tight')
print("Saved visualization to: iou_comparison.png")
plt.show()

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
ious = [m['iou'] for m in matches]
print(f"Mean IoU: {np.mean(ious):.4f}")
print(f"Min IoU:  {np.min(ious):.4f}")
print(f"Max IoU:  {np.max(ious):.4f}")
print(f"\nIndividual IoUs:")
for i, iou in enumerate(ious):
    print(f"  Object {i+1}: {iou:.4f}")