"""
This module launches a Gradio app for classifying images of Nigerian agricultural produce 
(beans, groundnut, maize, millet) using pre-trained ResNet and EfficientNet models.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
import torch
from PIL import Image
from models.resnet_model import get_resnet
from models.efficientnet_model import get_efficientnet
from utils.predictions import predict_image

# List of class names corresponding to produce categories
class_names = ['beans', 'groundnut', 'maize', 'millet']

# Load pre-trained ResNet model
resnet_model = get_resnet(len(class_names))
resnet_model.load_state_dict(torch.load('saved_models/best_resnet.pth', map_location='cpu'))
resnet_model.eval()

# Load pre-trained EfficientNet model
efficientnet_model = get_efficientnet(len(class_names))
efficientnet_model.load_state_dict(torch.load('saved_models/best_efficientnet.pth', map_location='cpu'))
efficientnet_model.eval()

def classify_image(img, model_name):
    """
    Classifies an input image using the selected model.

    Args:
        img (PIL.Image): The input image to classify.
        model_name (str): Model choice ("ResNet" or "EfficientNet").

    Returns:
        tuple: A dictionary containing predicted label and confidence, and the input image.
    """
    model = resnet_model if model_name == "ResNet" else efficientnet_model
    label, confidence = predict_image(model, img, class_names)
    return {"Predicted": label, "Confidence": f"{confidence:.2f}"}, img

# Define Gradio interface
gui = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(type="pil"),
        gr.Radio(choices=["ResNet", "EfficientNet"], label="Choose Model", value="ResNet")
    ],
    outputs=["json", "image"],
    title="Nigerian Produce Classifier"
)

def launch():
    """
    Launches the Gradio application with sharing enabled.
    """
    gui.launch(share=True)

if __name__ == '__main__':
    launch()
