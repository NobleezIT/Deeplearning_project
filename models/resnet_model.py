"""
This module provides a function to initialize a ResNet-50 model 
for image classification tasks with a custom number of output classes.
"""

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet(num_classes):
    """
    Initializes a ResNet-50 model with pretrained weights and a custom classification head.

    Args:
        num_classes (int): Number of output classes for classification.

    Returns:
        torch.nn.Module: A modified ResNet-50 model with the final fully connected layer adjusted.
    """
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the fully connected layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
