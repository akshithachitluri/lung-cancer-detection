# Lung Cancer Detection Using Deep Learning

## Project Overview
This project is a deep learning-based web application that detects different types of lung cancer from medical images using an EfficientNetB3 model.

The system classifies lung scan images into:
- Adenocarcinoma
- Large Cell Carcinoma
- Squamous Cell Carcinoma
- Normal

The project uses Flask as the backend API and HTML/CSS/JavaScript for the frontend interface.

---

## Technologies Used
- Python
- Flask
- TensorFlow / Keras
- EfficientNetB3
- NumPy
- Pillow
- HTML/CSS/JavaScript

---

## Features
- Upload lung scan images
- AI-based prediction
- Confidence score display
- Multiple cancer type classification
- Uncertain prediction handling
- Web-based interface

---

## Project Workflow
1. User uploads image
2. Flask backend receives image
3. Image preprocessing is applied
4. EfficientNet model predicts class
5. Prediction and confidence score are returned
6. Results displayed on frontend

---

## Model Details
The project uses EfficientNetB3 for feature extraction and classification.

Input image size:
224 x 224

Classes:
- adenocarcinoma
- large.cell.carcinoma
- normal
- squamous.cell.carcinoma

---

## Installation

Clone repository:

```bash
git clone https://github.com/akshithachitluri/lung-cancer-detectionit