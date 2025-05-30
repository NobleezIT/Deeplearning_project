# Nigerian Agricultural Produce Classifier 🌾

This is my submission for the Capstone project which is part of the requirements for the completion of the **ArewaDS DeepLearning Program Cohort 2**.

This project classifies images of Nigerian agricultural produce — namely **beans, groundnut, maize, and millet** — using deep learning models (ResNet and EfficientNet). It includes modular code for data loading, training, evaluation, and deployment via **Streamlit**.

---

## 📁 Project Structure

```

produce\_classifier/
│
├── app.py                      # Streamlit app interface
├── main.py                     # Entry script for training & evaluation
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── data/
│   └── split\_dataset/          # Training, validation, and test folders (see download instructions below)
│
├── models/
│   ├── resnet\_model.py
│   └── efficientnet\_model.py
│
├── saved\_models/               # Stores best model weights (.pth)
│
└── utils/
├── dataloader.py
├── train.py
├── evaluate.py
└── predictions.py

````

---

## 📦 Setup

To see the complete workflow, check my notebook [here](https://github.com/NobleezIT/Deeplearning_project/blob/main/Executionfile.ipynb).

1. **Clone the repository**:
```bash
git clone https://github.com/NobleezIT/Deeplearning_project.git
cd Deeplearning_project
````

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Download the dataset**:
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

## 🎨 Deployment with Streamlit

We provide a **Streamlit app** to interactively test the model.

1. **Install Streamlit** (if you haven’t already):

```bash
pip install streamlit
```

2. **Run the Streamlit app**:

```bash
streamlit run app.py
```

It will launch a web interface where you can upload images and see predictions!

---

## 📌 Notes

* All **model weights** are saved in `saved_models/`.
* **TensorBoard logs** are saved in the `logs/` directory.
* The app depends on the saved model weights (`best_resnet.pth` or `best_efficientnet.pth`).

---

