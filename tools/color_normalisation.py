# -*- coding: utf-8 -*-
"""color_normalisation.ipynb

"""

#!pip install histomicstk --find-links https://girder.github.io/large_image_wheels

import cv2, os
import pandas as pd
import girder_client
import numpy as np
from skimage.transform import resize
from matplotlib import pylab as plt
from matplotlib.colors import ListedColormap

######################

#!python --version

# load the required packages
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response)
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization
from histomicstk.preprocessing.color_deconvolution.\
    color_deconvolution import color_deconvolution_routine, stain_unmixing_routine
from histomicstk.preprocessing.augmentation.\
    color_augmentation import rgb_perturb_stain_concentration, perturb_stain_concentration

# color norm. standard (from TCGA-A2-A3XS-DX1, Amgad et al, 2019)
cnorm = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}

print(reinhard.__doc__)

#########################################################################################################
#program for color normalization#
#########################################################################################################
def ensure_dir(directory):
    """Ensure that a directory exists, and if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
        
def norm_and_save(img_path, output_folder):
    """Load an image, normalize it, and save it to the specified output folder."""
    # Ensure the output directory exists
    ensure_dir(output_folder)

    # Assuming img_path is correctly specified and img_norm is the normalized image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = reinhard(img_rgb, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'])
    img_norm_bgr = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
    
    # Construct the full output path for the image
    fname = os.path.basename(img_path)
    full_output_path = os.path.join(output_folder, fname)
    # Save the normalized image
    success = cv2.imwrite(full_output_path, img_norm)
    if success:
        print(f"Successfully saved normalized image to: {full_output_path}")
    else:
        print(f"Failed to save image: {full_output_path}")

def process_images(input_path, output_path):
    """Process all images in the input path and save them to the output path."""
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith('.png'):  # Check for PNG files
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_path)
                output_folder = os.path.join(output_path, relative_path)
                norm_and_save(img_path, output_folder)

if __name__ == "__main__":
    input_path = '/home/microcrispr8/Documents/project-oralcancer/ORCHID-2024/orchid-rename'
    output_path = '/home/microcrispr8/Documents/project-oralcancer/ORCHID-April24/ORCHID-norm'
    process_images(input_path, output_path)

