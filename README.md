# Plant Disease Classification with EfficientNet and Grad-CAM

This project implements a deep learning-based plant disease classification system using EfficientNet and Grad-CAM for model interpretability.

## Features
- Image-based plant disease classification (38 classes)
- Transfer learning with EfficientNet-B0
- Grad-CAM visualization for explainable AI
- Streamlit web application

## Requirements
- Python 3.9+
- PyTorch
- timm
- streamlit

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train model
python src/train.py

### 3. Run application
streamlit run app/app.py
