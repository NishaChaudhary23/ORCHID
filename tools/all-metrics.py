#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:00:57 2024

@author: microcrispr8
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Define base directory and fold numbers
base_dir = '/media/microcrispr8/DATA 1/ORCHID-April24/model-1/models/InceptionV3'
folds = range(1, 4)  # Assuming you have folds 1 to 3

# Loop through folds
for fold in folds:
    fold_dir = f"{base_dir}/InceptionV3_fold_{fold}"

    # Load data
    classification_report_df = pd.read_csv(f"{fold_dir}/Classification_report.csv")
    confusion_matrix_df = pd.read_csv(f"{fold_dir}/Confusion_matrix.csv", index_col=0)
    history_df = pd.read_csv(f"{fold_dir}/history.csv")

    # Calculate last accuracy values
    last_train_acc = history_df['acc'].iloc[-1] * 100
    last_val_acc = history_df['val_acc'].iloc[-1] * 100

    # Plotting accuracy graph
    plt.figure(figsize=(8, 6))
    plt.plot(history_df['acc'], label=f'Fold {fold} Train Acc ({last_train_acc:.2f}%)')
    plt.plot(history_df['val_acc'], label=f'Fold {fold} Val Acc ({last_val_acc:.2f}%)')
    plt.title(f'Accuracy for InceptionV3_fold_{fold}', fontsize=20, fontweight='bold', color='black')
    plt.ylabel('Accuracy', fontsize=16, fontweight='bold', color='black')
    plt.xlabel('Epoch', fontsize=16, fontweight='bold', color='black')
    plt.legend(loc='lower right')
    plt.xlim(0, 50)
    plt.xticks(range(0, 51, 10))
    plt.savefig(f"{fold_dir}/Accuracy_Plot.jpg", dpi=300)
    plt.close()  # Close plot to avoid overlapping

    # Plotting loss graph
    plt.figure(figsize=(8, 6))
    plt.plot(history_df['loss'], label='Fold {fold} Train Loss')
    plt.plot(history_df['val_loss'], label='Fold {fold} Val Loss')
    plt.title(f'Loss for InceptionV3_fold_{fold}', fontsize=20, fontweight='bold', color='black')
    plt.ylabel('Loss', fontsize=16, fontweight='bold', color='black')
    plt.xlabel('Epoch', fontsize=16, fontweight='bold', color='black')
    plt.legend(loc='upper right')
    plt.xlim(0, 50)
    plt.xticks(range(0, 51, 10))
    plt.savefig(f"{fold_dir}/Loss_Plot.jpg", dpi=300)
    plt.close()

    # Plotting confusion matrix heatmap
    # Convert raw counts to class-wise percentages
    row_sums = confusion_matrix_df.sum(axis=1).values.reshape(-1, 1)  # Sum of each row (true labels)
    confusion_matrix_per = (confusion_matrix_df / row_sums * 100).round(2)  # Normalize each row by its sum and convert to percentage

    # Create a new DataFrame for annotations combining counts and percentages
    annot_labels = (confusion_matrix_df.astype(str) + "\n(" + confusion_matrix_per.astype(str) + "%)").values

    cubehelix_cm = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_df, annot=annot_labels, fmt='', cmap=cubehelix_cm, annot_kws={"size": 14, "weight": "bold"})
    plt.title(f'Confusion Matrix for InceptionV3_fold_{fold}', fontsize=20, fontweight='bold', color='black')
    plt.xticks(color='black', fontsize=16)
    plt.yticks(color='black', fontsize=16)
    plt.ylabel('True Label', fontsize=16, fontweight='bold', color='black')
    plt.xlabel('Predicted Label', fontsize=16, fontweight='bold', color='black')
    plt.savefig(f"{fold_dir}/Confusion_Matrix.jpg", dpi=300)
    plt.close()
    
print("Plots generated for all folds!")



###############################################################################################
#Plot metrics for test data
# Define base directory and fold numbers
base_dir = '/media/microcrispr8/DATA 1/ORCHID-April24/model-2/model_test_output'
folds = range(1, 4)  # Assuming you have folds 1 to 3

# Loop through folds
for fold in folds:
    fold_dir = f"{base_dir}/InceptionV3_fold_{fold}"

    # Load data
    confusion_matrix_df = pd.read_csv(f"{fold_dir}/Confusion_matrix.csv", index_col=0)
    
    # Convert raw counts to class-wise percentages
    row_sums = confusion_matrix_df.sum(axis=1).values.reshape(-1, 1)  # Sum of each row (true labels)
    confusion_matrix_per = (confusion_matrix_df / row_sums * 100).round(2)  # Normalize each row by its sum and convert to percentage

    # Create a new DataFrame for annotations combining counts and percentages
    annot_labels = (confusion_matrix_df.astype(str) + "\n(" + confusion_matrix_per.astype(str) + "%)").values

    cubehelix_cm = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_df, annot=annot_labels, fmt='', cmap=cubehelix_cm, annot_kws={"size": 14, "weight": "bold"})
    plt.title(f'Confusion Matrix for InceptionV3_fold_{fold}', fontsize=20, fontweight='bold', color='black')
    plt.xticks(color='black', fontsize=16)
    plt.yticks(color='black', fontsize=16)
    plt.ylabel('True Label', fontsize=16, fontweight='bold', color='black')
    plt.xlabel('Predicted Label', fontsize=16, fontweight='bold', color='black')
    plt.savefig(f"{fold_dir}/Confusion_Matrix.jpg", dpi=300)
    plt.close()












