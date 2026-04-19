"""Evaluation utilities: mAP, classification report, confusion matrix.

Provides functions to compute standard classification metrics
and visualize model performance across all disease classes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize


def evaluate_model(model, val_loader, label_map, device=None):
    """Computes accuracy, mAP and per-class classification metrics.

    Runs the model on the full validation set, collecting predicted
    probabilities and true labels to compute metrics.

    Args:
        model (nn.Module): Trained model in eval mode.
        val_loader (DataLoader): Validation data loader.
        label_map (dict): Mapping from disease name (str) to class index (int).
        device (torch.device, optional): Device to run inference on.
            Defaults to GPU if available.

    Returns:
        tuple: (acc, mAP, all_probs, all_labels)
            - acc (float): Overall accuracy (correct / total).
            - mAP (float): Mean Average Precision across all classes present in val set.
            - all_probs (np.ndarray): Predicted probabilities, shape (N, num_classes).
            - all_labels (np.ndarray): True labels, shape (N,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reverse mapping: index to disease name for readable output
    idx_to_disease = {v: k for k, v in label_map.items()}
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            # Convert logits to probabilities with softmax
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs) #(N, 39)
    all_labels = np.concatenate(all_labels) #(N,)
    preds = all_probs.argmax(axis=1) #predicted class per sample

    # mAP
    y_true_bin = label_binarize(all_labels, classes=range(len(label_map)))
    per_class_ap = []
    for i in range(len(label_map)):
        if y_true_bin[:, i].sum() > 0: #skip classes with no val samples
            per_class_ap.append(
                average_precision_score(y_true_bin[:, i], all_probs[:, i])
            )
    mAP = np.mean(per_class_ap)

    acc = (preds == all_labels).mean()

    print(f"Accuracy: {acc:.4f} | mAP: {mAP:.4f}")
    print(
        classification_report(
            all_labels,
            preds,
            target_names=[idx_to_disease[i] for i in range(len(label_map))],
            zero_division=0,
        )
    )

    return acc, mAP, all_probs, all_labels


def plot_confusion_matrix(all_probs, all_labels, label_map, save_path=None):
    """Plots a heatmap confusion matrix for all disease classes.

    Args:
        all_probs (np.ndarray): Predicted probabilities from evaluate_model(),
            shape (N, num_classes).
        all_labels (np.ndarray): True labels from evaluate_model(), shape (N,).
        label_map (dict): Disease name to index mapping.
        save_path (str, optional): If given, saves the plot as an image file.
    """
    idx_to_disease = {v: k for k, v in label_map.items()}
    preds = all_probs.argmax(axis=1)
    cm = confusion_matrix(all_labels, preds)
    disease_names = [idx_to_disease[i] for i in range(len(label_map))]

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm,
        annot=True, #show count in each cell
        fmt="d", #integer format
        cmap="Blues",
        xticklabels=disease_names,
        yticklabels=disease_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
