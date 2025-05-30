import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import os

from models.resnet_model import get_resnet
from models.efficientnet_model import get_efficientnet
# from utils.dataloader import get_class_names

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
MODEL_PATHS = {
    "ResNet": "saved_models/best_resnet.pth",
    "EfficientNet": "saved_models/best_efficientnet.pth"
}
# DATA_DIR = "data/split_dataset"
IMG_SIZE = 224

# Transform definition
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model and class names
@st.cache_resource
def load_model(model_name, num_classes):
    if model_name == "ResNet":
        model = get_resnet(num_classes)
    else:
        model = get_efficientnet(num_classes)
    model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location=device))
    model.to(device)
    model.eval()
    return model

# Load class names
# class_names = get_class_names(DATA_DIR)

class_names = ['beans', 'groundnut', 'maize', 'millet']


# UI
st.title("🧠 Nigerian Produce Classifier")
st.markdown("Upload an image of **Beans**, **Groundnut**, **Maize**, or **Millet** to classify it.")

model_choice = st.selectbox("Choose Model", ["ResNet", "EfficientNet"])
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    model = load_model(model_choice, len(class_names))
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, 1)

    st.success(f"🎯 Prediction: **{class_names[prediction.item()].capitalize()}**")
    st.info(f"Confidence: `{confidence.item() * 100:.2f}%`")
