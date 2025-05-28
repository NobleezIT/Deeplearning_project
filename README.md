
# Nigerian Agricultural Produce Classifier 🌾
This my submission for the Capstone project which is part of the requirement for the completion of the **ArewaDs DeepLearning Program Cohort 2**. 
This project classifies images of Nigerian agricultural produce, namely: **beans, groundnut, maize, and millet**, using deep learning models (ResNet and EfficientNet). It includes modular code for data loading, training, evaluation, and deployment via Gradio.

---

## 📁 Project Structure

```
produce_classifier/
│
├── app.py                      # Gradio app interface
├── main.py                     # Entry script for training & evaluation
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── data/
│   └── split_dataset/          # Training, validation, and test folders (see download instructions below)
│
├── models/
│   ├── resnet_model.py
│   └── efficientnet_model.py
│
├── saved_models/               # Stores best model weights (.pth)
│
└── utils/
    ├── dataloader.py
    ├── train.py
    ├── evaluate.py
    └── predictions.py
```

---

## 📦 Setup

1. **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/produce_classifier.git
cd produce_classifier
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download the dataset**

Download the split dataset from [Google Drive](https://drive.google.com/file/d/1Q2VE5lTBEd_-0BnVUBZLatOYqc-WDznn/view?usp=sharing):

* Go to the link above.
* Click **Download**.
* Extract the ZIP file.
* Move the extracted folder to `data/split_dataset/` so your structure looks like:

```
produce_classifier/
└── data/
    └── split_dataset/
        ├── train/
        ├── val/
        └── test/
```

---

## 🚀 Training & Evaluation

Run the following command to train and evaluate:

```bash
python main.py
```

Customize training:

```bash
python main.py --model efficientnet --epochs 15 --batch_size 64 --lr 0.0005 --data_dir data/split_dataset
```

---

## 🖼️ Gradio App (Local Deployment)

```bash
python app.py
```

It will provide a public link you can click on to view and test the application.

---

## 📌 Notes

* All model weights are saved to `saved_models/`
* TensorBoard logs are saved in the `logs/` directory

---
