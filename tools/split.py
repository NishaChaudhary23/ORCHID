#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:33:50 2024

@author: microcrispr8
"""

import os
import shutil
import random

def split_data(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    # Define classes and split types
    classes = ['normal', 'osmf', 'wdoscc', 'mdoscc', 'pdoscc']
    splits = ['train', 'val', 'test']

    # Create output directories mirroring the input structure
    for split in splits:
        for class_name in classes:
            class_path = os.path.join(source_dir, class_name)
            patients = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
            for patient in patients:
                os.makedirs(os.path.join(output_dir, split, class_name, patient), exist_ok=True)

    # Process each class and each patient folder
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        patients = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]

        for patient in patients:
            patient_path = os.path.join(class_path, patient)
            images = [f for f in os.listdir(patient_path) if f.endswith('.png')]
            random.shuffle(images)  # Randomize image order

            # Determine split counts
            n_total = len(images)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            n_test = n_total - n_train - n_val

            # Assign images to train, val, and test sets
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]

            # Copy files to the corresponding subdirectories
            def copy_files(files, split):
                for file in files:
                    src_path = os.path.join(patient_path, file)
                    dest_path = os.path.join(output_dir, split, class_name, patient, file)
                    shutil.copy(src_path, dest_path)

            copy_files(train_images, 'train')
            copy_files(val_images, 'val')
            copy_files(test_images, 'test')

# Usage
source_directory = '/home/microcrispr8/Documents/project-oralcancer/ORCHID-April24/ORCHID-norm'
output_directory = '/home/microcrispr8/Documents/project-oralcancer/ORCHID-April24/split-data'
split_data(source_directory, output_directory)
