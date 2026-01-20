import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
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

gt_df = pd.read_csv('new_output/gnd_truth_plain_bbox.csv')
seg_df = pd.read_csv('new_output/segmented_image_bbox.csv')
# seg_df = pd.read_csv('new_output/sam_iamge_bbox.csv')

# Get image dimensions
img_width = gt_df.iloc[0]['image_width']
img_height = gt_df.iloc[0]['image_height']

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_xlim(0, img_width)
ax.set_ylim(img_height, 0)
ax.set_aspect('equal')

# Match boxes and calculate IoUs for labels
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
        matches.append((gt_idx, best_seg_idx, best_iou))

# Draw all ground truth boxes (GREEN)
for idx, gt_box in gt_df.iterrows():
    rect = patches.Rectangle(
        (gt_box['bbox_x'], gt_box['bbox_y']),
        gt_box['bbox_width'],
        gt_box['bbox_height'],
        linewidth=3,
        edgecolor='green',
        facecolor='none',
        label='Ground Truth' if idx == 0 else ''
    )
    ax.add_patch(rect)
    
    # Add label for ground truth
    ax.text(gt_box['bbox_x'], gt_box['bbox_y'] - 5, 
            f"GT: {gt_box['label_name']}", 
            color='green', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Draw all segmented boxes (RED)
for idx, seg_box in seg_df.iterrows():
    rect = patches.Rectangle(
        (seg_box['bbox_x'], seg_box['bbox_y']),
        seg_box['bbox_width'],
        seg_box['bbox_height'],
        linewidth=3,
        edgecolor='red',
        facecolor='none',
        linestyle='--',
        label='Segmented' if idx == 0 else ''
    )
    ax.add_patch(rect)
    
    # Add label for segmented
    ax.text(seg_box['bbox_x'] + seg_box['bbox_width'], 
            seg_box['bbox_y'] - 5, 
            f"Seg: {seg_box['label_name']}", 
            color='red', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            ha='right')

# Add IoU values at the center of matched pairs
for gt_idx, seg_idx, iou in matches:
    gt_box = gt_df.iloc[gt_idx]
    seg_box = seg_df.iloc[seg_idx]
    
    # Calculate center point
    center_x = (gt_box['bbox_x'] + gt_box['bbox_x'] + gt_box['bbox_width'] + 
                seg_box['bbox_x'] + seg_box['bbox_x'] + seg_box['bbox_width']) / 4
    center_y = (gt_box['bbox_y'] + gt_box['bbox_y'] + gt_box['bbox_height'] + 
                seg_box['bbox_y'] + seg_box['bbox_y'] + seg_box['bbox_height']) / 4
    
    # Color code by IoU quality
    if iou >= 0.75:
        iou_color = 'darkgreen'
    elif iou >= 0.5:
        iou_color = 'orange'
    else:
        iou_color = 'red'
    
    ax.text(center_x, center_y, f"IoU: {iou:.3f}", 
            color=iou_color, fontsize=12, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                     edgecolor=iou_color, linewidth=2, alpha=0.9))

# Calculate mean IoU
ious = [m[2] for m in matches]
mean_iou = np.mean(ious)

# Add title and legend
ax.set_title(f'Ground Truth vs Opencv\nMean IoU: {mean_iou:.4f}', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('X (pixels)', fontsize=12)
ax.set_ylabel('Y (pixels)', fontsize=12)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig('bbox_overlay_cvvgnd.png', dpi=150, bbox_inches='tight')
print("Saved overlay image to: bbox_overlay_comparison.png")
# plt.savefig('bbox_overlay_samvgnd.png', dpi=150, bbox_inches='tight')
# print("Saved overlay image to: bbox_overlay_samvgnd.png")
plt.show()

print("\n")
print("OVERLAY COMPARISON \n")
print(f"Mean IoU: {mean_iou:.4f}")
print(f"Total matches: {len(matches)}")
print("\nIndividual IoUs:")
for gt_idx, seg_idx, iou in matches:
    print(f"  {gt_df.iloc[gt_idx]['label_name']:10s} -> "
          f"{seg_df.iloc[seg_idx]['label_name']:10s}  IoU: {iou:.4f}")