"""
Module for evaluating a trained image classification model using accuracy metrics,
classification report, and confusion matrix.

Includes support for both ResNet and EfficientNet architectures.
"""

import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from models.resnet_model import get_resnet
from models.efficientnet_model import get_efficientnet

def evaluate_model(model_path, test_loader, class_names):
    """
    Evaluates a trained model on a given test dataset and prints classification metrics.

    Args:
        model_path (str): Path to the saved model file (.pth).
        test_loader (DataLoader): PyTorch DataLoader for the test dataset.
        class_names (list): List of class names corresponding to target labels.

    Raises:
        ValueError: If the model type cannot be determined from the model path.

    Prints:
        - Classification report (precision, recall, f1-score).
        - Confusion matrix of predictions vs. true labels.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine model type from model path
    if 'resnet' in model_path:
        model = get_resnet(len(class_names))
    elif 'efficientnet' in model_path:
        model = get_efficientnet(len(class_names))
    else:
        raise ValueError("Unknown model type in path.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=class_names))
    print(confusion_matrix(y_true, y_pred))
