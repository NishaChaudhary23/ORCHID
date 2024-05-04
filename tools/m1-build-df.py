#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 19:37:28 2024

@author: microcrispr8
"""

import os
import pandas as pd
import numpy as np
import sys

def build_df(base_path, output_path, setseed=42, data_type=1):
    # Set the random seed for reproducibility
    np.random.seed(setseed)
    
    # Directories to process as a combined set
    dirs_to_process = ['train', 'val']
    labels = ['oscc', 'normal', 'osmf']  # Top level labels
    subtype_labels = ['wdoscc', 'mdoscc', 'pdoscc'] # Subtypes within 'OSCC'
    
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
            label_path = os.path.join(dir_path, label_dir)
            if label_dir in labels:
                process_directory(label_path, label_dir, dir_name, data, subtype_labels, data_type)

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

def process_directory(path, label, split, data, subtype_labels, data_type):
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)
        if os.path.isdir(entry_path):
            # Check for subtypes in 'OSCC' folder
            if label == 'oscc' and any(entry.startswith(prefix) for prefix in subtype_labels):
                # Label all subtypes as 'OSCC' if data_type is set accordingly
                if data_type == 2:
                    process_directory(entry_path, 'oscc', split, data, subtype_labels, data_type)
                else:
                    specific_label = entry[:2] + 'oscc'  # Construct label like 'WDOSCC'
                    process_directory(entry_path, specific_label, split, data, subtype_labels, data_type)
            else:
                process_directory(entry_path, label, split, data, subtype_labels, data_type)
        elif entry.endswith('.png'):
            data['filename'].append(entry_path)
            data['label'].append(label)
            data['split'].append(split)
            
if __name__ == "__main__":
    base_path = '/media/microcrispr8/DATA 1/ORCHID-April24/model-1'
    output_path = '/media/microcrispr8/DATA 1/ORCHID-April24/model-1'
    data_type = 2  # Change as needed for different data types
    build_df(base_path, output_path, data_type=data_type)
