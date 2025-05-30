"""
Module for loading a trained image classification model and making a prediction on a single image.

Supports both ResNet and EfficientNet architectures.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from torchvision import transforms
from PIL import Image
from models.resnet_model import get_resnet
from models.efficientnet_model import get_efficientnet
from utils.dataloader import get_class_names


def load_model(model_type, model_path, num_classes):
    """
    Loads a trained model from the specified path.

    Args:
        model_type (str): Type of model ('resnet' or 'efficientnet').
        model_path (str): Path to the trained model file (.pth).
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.
    """
    if model_type == 'resnet':
        model = get_resnet(num_classes)
    else:
        model = get_efficientnet(num_classes)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict_image(model, image, class_names):
    """
    Predicts the class of a given image using the provided model.

    Args:
        model (torch.nn.Module): The trained model for prediction.
        image (PIL.Image.Image): The input image.
        class_names (list): List of class names.

    Returns:
        tuple: Predicted class label and confidence score.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()

    return class_names[predicted_index], confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['resnet', 'efficientnet'],
                        help='Model type to use for prediction.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file.')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the image file to classify.')
    parser.add_argument('--data_dir', type=str, default='data/split_dataset',
                        help='Directory containing the training data to infer class names.')
    args = parser.parse_args()

    class_names = get_class_names(args.data_dir)
    model = load_model(args.model, args.model_path, len(class_names))
    image = Image.open(args.image_path).convert('RGB')
    label, confidence = predict_image(model, image, class_names)

    print(f"Predicted Label: {label}")
    print(f"Confidence: {confidence:.4f}")
