# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def read_pgm(file_path):
#     return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# def find_zero_coordinates(img):
#     # Find zero-value pixels
#     coordinates = np.column_stack(np.where(img == 0))
#     return coordinates

# def plot_zero_coordinates_superimposed(img1, img2, output_file):
#     zero_coords1 = find_zero_coordinates(img1)
#     zero_coords2 = find_zero_coordinates(img2)
    
#     plt.figure(figsize=(8, 8))
    
#     if zero_coords1.size > 0:
#         plt.plot(zero_coords1[:, 1], img1.shape[0] - zero_coords1[:, 0], 'r.', markersize=1, label='Real Map')

#     if zero_coords2.size > 0:
#         plt.plot(zero_coords2[:, 1], img2.shape[0] - zero_coords2[:, 0], 'b.', markersize=1, label='Virtual Map')

#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.title('Superimposed Zero-value Pixels')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.legend(bbox_to_anchor =(0.6, 0), ncol = 1)
#     plt.tight_layout()
#     plt.axis('off')
    
#     # Save the plot as an image file
#     plt.savefig(output_file, format='png')
#     plt.close()

# def process_folder(folder_real, folder_virtual, output_folder):
#     # Ensure output folder exists
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # List all files in the real map folder
#     real_files = sorted(os.listdir(folder_real))
#     virtual_files = sorted(os.listdir(folder_virtual))

#     # Traverse through each file pair
#     for real_file, virtual_file in zip(real_files, virtual_files):
#         real_path = os.path.join(folder_real, real_file)
#         virtual_path = os.path.join(folder_virtual, virtual_file)

#         if real_file.endswith('.pgm') and virtual_file.endswith('.pgm'):
#             # Read images
#             real_img = read_pgm(real_path)
#             virtual_img = read_pgm(virtual_path)
            
#             # Define output file path
#             output_file = os.path.join(output_folder, f"superimposed_{real_file.split('.')[0]}.png")
            
#             # Plot and save the superimposed image
#             plot_zero_coordinates_superimposed(real_img, virtual_img, output_file)
#             print(f"Saved superimposed plot for {real_file} and {virtual_file} at {output_file}")


# folder_real_maps = 'new_r_maps'
# folder_virtual_maps = 'new_v_maps'
# output_folder = 'new_Map_matching_ops'

# process_folder(folder_real_maps, folder_virtual_maps, output_folder)



##################################### old code #####################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_pgm(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def find_zero_coordinates(img):
    coordinates = np.column_stack(np.where(img == 0))
    return coordinates

def plot_zero_coordinates_superimposed(img1, img2):
    zero_coords1 = find_zero_coordinates(img1)
    zero_coords2 = find_zero_coordinates(img2)
    
    plt.figure(figsize=(8, 8))
    if zero_coords1.size > 0:
        plt.plot(zero_coords1[:, 1], img1.shape[0] - zero_coords1[:, 0], 'r.', markersize=1, label='Real Map')

    if zero_coords2.size > 0:
        plt.plot(zero_coords2[:, 1], img2.shape[0] - zero_coords2[:, 0], 'b.', markersize=1, label='Ground Truth')
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Superimposed Zero-value Pixels')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(bbox_to_anchor =(0.6, 0), ncol = 1)
    plt.tight_layout()
    plt.axis('off')
    plt.show()

image1_real = read_pgm('old_r_maps/map_r_10_11_16.pgm')
image2_virtual = read_pgm('old_v_maps/map_v_10_12_26.pgm')

plot_zero_coordinates_superimposed(image1_real, image2_virtual)