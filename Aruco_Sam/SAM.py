import torch
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

def select_points(image, existing_points=None):

    points = existing_points.tolist() if isinstance(existing_points, np.ndarray) else (existing_points or [])

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            points.append((int(event.xdata), int(event.ydata)))
            print(f"Selected point: {points[-1]}")
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()

    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.title("Click on the object to select points. Close the window when done.")
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if not points:
        print("No points were selected. Please try again.")

    return np.array(points)

def combine_selected_masks(masks, selected_masks):
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    
    for mask_index in selected_masks:
        if np.any(masks[mask_index]):
            combined_mask = np.maximum(combined_mask, masks[mask_index])
    
    return combined_mask


def display_masks(masks, scores, image_rgb):
    for i, mask in enumerate(masks):
        overlay = image_rgb.copy()
        overlay[mask > 0] = [255, 0, 0]
        plt.figure()
        plt.title(f"Mask {i + 1} - Score: {scores[i]:.2f}")
        plt.imshow(overlay)
        plt.axis("off")
        plt.show()
        
def refine_mask(image_rgb, predictor, masks, scores, input_points, selected_masks):
    while True:
        print("Options:")
        print("1. Add a mask to the selected mask list.")
        print("2. Remove a mask from the selected mask list.")
        print("3. Add new points to create a new mask.")
        print("4. Save a specific mask to a file.")
        print("5. Finalize and save combined results.")
        choice = input("Enter your choice (1/2/3/4/5): ").strip()

        if choice == "1":
            try:
                mask_index = int(input(f"Select a mask to add to the list (1-{len(masks)}): ").strip()) - 1
                if 0 <= mask_index < len(masks):
                    if mask_index not in selected_masks:
                        selected_masks.append(mask_index)
                        print(f"Mask {mask_index + 1} added to the selected mask list.")
                    else:
                        print(f"Mask {mask_index + 1} is already in the selected mask list.")
                else:
                    print("Invalid mask number.")
            except ValueError:
                print("Invalid input. Please enter a valid mask number.")

        elif choice == "2":
            try:
                mask_index = int(input(f"Select a mask to remove from the list (1-{len(masks)}): ").strip()) - 1
                if mask_index in selected_masks:
                    selected_masks.remove(mask_index)
                    print(f"Mask {mask_index + 1} removed from the selected mask list.")
                else:
                    print(f"Mask {mask_index + 1} is not in the selected mask list.")
            except ValueError:
                print("Invalid input. Please enter a valid mask number.")

        elif choice == "3":
            new_points = select_points(image_rgb)
            if len(new_points) == 0:
                print("No points selected. Returning to menu.")
                continue

            new_labels = np.ones(len(new_points))
            predictor.set_image(image_rgb)
            new_masks, new_scores, _ = predictor.predict(
                point_coords=new_points,
                point_labels=new_labels,
                multimask_output=True
            )

            masks = np.concatenate([masks, new_masks], axis=0)
            scores = np.concatenate([scores, new_scores], axis=0)

            print(f"Generated {len(new_masks)} new masks. Total masks: {len(masks)}.")
            display_masks(new_masks, new_scores, image_rgb)

        elif choice == "4":
            print("Displaying masks for selection:")
            for i, mask in enumerate(masks):
                overlay = image_rgb.copy()
                overlay[mask > 0] = [255, 0, 0]
                plt.figure()
                plt.title(f"Mask {i + 1} - Score: {scores[i]:.2f}")
                plt.imshow(overlay)
                plt.axis("off")
                plt.show()

            try:
                mask_index = int(input(f"Select a mask to save (1-{len(masks)}): ").strip()) - 1
                if 0 <= mask_index < len(masks):
                    path = f"new_output/mask_{mask_index + 1}.png"
                    cv2.imwrite(path, (masks[mask_index] * 255).astype("uint8"))
                    print(f"Saved Mask {mask_index + 1} to {path}.")
                    selected_masks.append((mask_index) + 1)
                else:
                    print("Invalid mask number.")
            except ValueError:
                print("Invalid input. Please enter a valid mask number.")


        elif choice == "5":
            print("Combining all saved masks...")

            combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
            print(selected_masks)
            for mask_index in selected_masks:
                if mask_index >= len(masks) or mask_index < 0:
                    print(f"Invalid mask index: {mask_index}. Skipping.")
                    continue

                print(f"Combining mask {mask_index} with shape: {masks[mask_index].shape}")

                combined_mask = np.maximum(combined_mask, masks[mask_index])
          
            final_mask_path = " new_output/final_combined_mask.png"
            cv2.imwrite(final_mask_path, (combined_mask * 255).astype("uint8"))
            print(f"Saved final combined mask to {final_mask_path}.")

            plt.figure()
            plt.title("Final Combined Mask")
            plt.imshow(combined_mask, cmap="gray")
            plt.axis("off")
            plt.show()
 


def save_results(masks, selected_masks, final_mask_path):
    """Saves combined selected masks and outputs the final result."""
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for i in selected_masks:
        combined_mask = np.maximum(combined_mask, masks[i])

    cv2.imwrite(final_mask_path, (combined_mask * 255).astype("uint8"))
    print(f"Saved final combined mask to {final_mask_path}")

def main():
    model_type = "vit_l"
    checkpoint_path = "aruco_sam/sam_vit_l_0b3195.pth"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)

    image_path = "aruco_sam/Transformed and Processed Image.png"
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    print("Select initial points on the image.")
    input_points = select_points(image_rgb)

    if input_points.size == 0:
        print("No points selected. Exiting.")
        return

    input_labels = np.ones(len(input_points))
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    display_masks(masks, scores, image_rgb)

    print("Refining masks and managing selected masks...")
    selected_masks = []
    final_masks, selected_masks = refine_mask(image_rgb, predictor, masks, scores, input_points, selected_masks)

    final_mask_path = "new_output/final_combined_mask.png"
    save_results(final_masks, selected_masks, final_mask_path)

if __name__ == "__main__":
    main()
