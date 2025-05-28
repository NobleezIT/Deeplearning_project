"""
This module provides a function to initialize an EfficientNet-B0 model 
for image classification tasks with a custom number of output classes.
"""

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_efficientnet(num_classes):
    """
    Initializes an EfficientNet-B0 model with pretrained weights and custom classifier.

    Args:
        num_classes (int): Number of output classes for classification.

    Returns:
        torch.nn.Module: A modified EfficientNet-B0 model with updated classifier layer.
    """
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    
    # Freeze all feature extraction layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classification layer
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model
