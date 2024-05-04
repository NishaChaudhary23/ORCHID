##Patch generation, patch size=300, overlap=150

import os
from PIL import Image

# Define the input directory where the images are located, with multiple subfolders
input_directory = '/home/microcrispr8/Documents/project-oralcancer/ORCHID-April24/split-data'

# Define the output directory where the image patches will be saved
output_directory = '/home/microcrispr8/Documents/project-oralcancer/ORCHID-April24/patches-with128overlap'

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def generate_and_save_patches(image_path, output_dir, patch_size=(512, 512), overlap=256):
    with Image.open(image_path) as img:
        img_width, img_height = img.size
        
        # Calculate the number of patches along width and height
        x_steps = (img_width - patch_size[0]) // (patch_size[0] - overlap) + 1
        y_steps = (img_height - patch_size[1]) // (patch_size[1] - overlap) + 1
        
        # Generate patches
        for x in range(x_steps):
            for y in range(y_steps):
                top_left_x = x * (patch_size[0] - overlap)
                top_left_y = y * (patch_size[1] - overlap)
                bottom_right_x = top_left_x + patch_size[0]
                bottom_right_y = top_left_y + patch_size[1]

                # Ensure the patch is within image bounds
                if bottom_right_x <= img_width and bottom_right_y <= img_height:
                    patch = img.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
                    output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_patch_{x}_{y}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    patch.save(output_path)
                    print(f"Saved patch: {output_path}")

# Function to iterate through subfolders and process images
def process_subfolders(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.png'):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                subfolder_output_dir = os.path.join(output_dir, relative_path)
                
                if not os.path.exists(subfolder_output_dir):
                    os.makedirs(subfolder_output_dir)
                
                generate_and_save_patches(image_path, subfolder_output_dir)

# Start the processing
process_subfolders(input_directory, output_directory)
