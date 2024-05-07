#Testing model-1
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

def process_directory(path, label, subtype_labels, data_type):
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)
        if os.path.isdir(entry_path):
            if label == 'oscc' and any(entry.lower().startswith(prefix) for prefix in subtype_labels):
                if data_type == 2:
                    process_directory(entry_path, 'oscc', subtype_labels, data_type)
                else:
                    specific_label = entry[:2].upper() + 'OSCC'  # Construct label like 'WDOSCC'
                    process_directory(entry_path, specific_label, subtype_labels, data_type)
            else:
                process_directory(entry_path, label, subtype_labels, data_type)

def evaluate_model(model_path, test_data_dir, output_dir, class_names):
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(512, 512),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

    print("Evaluating the model...")
    results = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

    print("Generating confusion matrix...")
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    confusion_mtx = confusion_matrix(test_generator.classes, y_pred)
    # getting dataloader keys
    conf_df = pd.DataFrame(confusion_mtx, index = list(test_generator.class_indices.keys()), columns = list(test_generator.class_indices.keys()))
    conf_df.to_csv(os.path.join(output_dir, 'Confusion_matrix.csv'))
    
    # plot cm
    cubehelix_cm = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_df, annot=True, fmt='d', cmap=cubehelix_cm)
    plt.title('Confusion Matrix', fontsize=20, fontweight='bold', color='black')
    plt.xticks(color='black', fontsize=16)
    plt.yticks(color='black', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16, fontweight='bold', color='black')
    plt.ylabel('True Label', fontsize=16, fontweight='bold', color='black')
    plt.savefig(os.path.join(output_dir, 'Confusion_matrix.jpg'))
    
    # Computing and saving the Classification Report
    # classification report
    target_names = list(test_generator.class_indices.keys())
    report = classification_report(test_generator.classes, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(output_dir,'Classification_report.csv'))
    
    # Saving predicted labels
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_labels_df = pd.DataFrame({'Image': test_generator.filenames, 'Predicted Label': predicted_labels})
    predicted_labels_df.to_csv(os.path.join(output_dir, 'Predicted_labels.csv'), index=False)

    # ROC curves
    n_classes = len(class_names)  #  3 classes 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        y_true_i = test_generator.classes == i
        y_score_i = predictions[:, i]
        fpr[i], tpr[i], _ = roc_curve(y_true_i, y_score_i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC Curve of {class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves', fontsize=20, fontweight='bold', color='black')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'ROC_Curve.jpg'))
    plt.close()


if __name__ == "__main__":
    base_model_dir = '/media/microcrispr8/DATA 1/ORCHID-April24/model-1/models/InceptionV3'
    model_folders = ['InceptionV3_fold_1', 'InceptionV3_fold_2', 'InceptionV3_fold_3']
    test_data_directory = '/media/microcrispr8/DATA 1/ORCHID-April24/test/model-1-test-data'
    output_base_dir = '/media/microcrispr8/DATA 1/ORCHID-April24/model-1/model_test_output'
    class_names = ['Normal', 'OSCC', 'OSMF'] 
    subtype_labels = ['wd', 'md', 'pd']  # Initials of the subtype names

    for folder in model_folders:
        model_dir = os.path.join(base_model_dir, folder)
        output_dir = os.path.join(output_base_dir, folder)
        model_file = os.path.join(model_dir, folder + '.keras')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Starting evaluation for {folder}...")
        evaluate_model(model_file, test_data_directory, output_dir, class_names)
        print(f"Finished evaluation for {folder}.\n")
