#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 18:22:25 2024

@author: microcrispr8
"""

import os
import pandas as pd
import numpy as np
import sys

def build_df(base_path, output_path, setseed=42):
    # Set the random seed for reproducibility
    np.random.seed(setseed)
    
    # Directories to process as a combined set
    dirs_to_process = ['train', 'val']
    #labels = ['oscc', 'normal', 'osmf']
    labels = ['wdoscc', 'mdoscc', 'pdoscc']
    
    # Create save path if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = {
        'filename': [],
        'label': [],
        'split': []
    }

    # Process each main directory (train, val)
    for dir_name in dirs_to_process:
        dir_path = os.path.join(base_path, dir_name)
        for label_dir in os.listdir(dir_path):
            if label_dir in labels:  # Check if the directory is one of the labels
                label_path = os.path.join(dir_path, label_dir)
                for subfolder in os.listdir(label_path):  # Access subfolders within each label directory
                    subfolder_path = os.path.join(label_path, subfolder)
                    for file in os.listdir(subfolder_path):  # Iterate through files in subfolders
                        if file.endswith('.png'):
                            filepath = os.path.join(subfolder_path, file)
                            data['filename'].append(filepath)
                            data['label'].append(label_dir)
                            data['split'].append(dir_name)

    # Create DataFrame and shuffle it
    df = pd.DataFrame(data)
    df_shuffled = df.sample(frac=1, random_state=setseed).reset_index(drop=True)

    # Split the data into 3 folds
    total_length = len(df_shuffled)
    fold_size = total_length // 3
    for i in range(3):
        fold_df = df_shuffled.iloc[i*fold_size : (i+1)*fold_size if i<2 else None]  # Ensure the last fold includes any remaining data
        fold_filename = os.path.join(output_path, f"dataset_fold_{i+1}.csv")
        fold_df.to_csv(fold_filename, index=False)

    print("DataFrames for each fold have been saved.")

if __name__ == "__main__":
    # Hardcoded paths for demonstration
    base_path = '/media/microcrispr8/DATA 1/ORCHID-April24/model-2'
    output_path = '/media/microcrispr8/DATA 1/ORCHID-April24/model-2'
    # Run the function with the paths
    build_df(base_path, output_path)
