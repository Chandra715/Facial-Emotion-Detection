# Face Emotion Recognition

A deep learning project for multi-class facial emotion recognition using TensorFlow/Keras. This project trains a Convolutional Neural Network (CNN) on grayscale facial images and classifies expressions into seven emotion categories.

## Project Overview

This project builds an end-to-end image classification pipeline for facial emotion recognition. It includes:

- data loading from directory-based train/test folders
- CNN-based model training with batch normalization and dropout
- model checkpointing
- training/validation metric visualization
- confusion matrix evaluation
- export of trained model architecture and weights

## Problem Statement

Facial emotion recognition is a computer vision task that predicts the emotional state expressed in an image. This project is designed to classify images into one of seven emotion classes, which is commonly aligned with FER-style datasets.

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn

## Suggested Dataset Structure

Place your dataset like this inside the `data/` directory:

```text
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## How to Run

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py --data_dir data --epochs 50 --batch_size 128
```

### 4. Outputs

Training generates these artifacts:

- `models/best_model.weights.h5`
- `models/emotion_model.json`
- `models/training_history.png`
- `models/confusion_matrix.png`

## Recruiter-Focused Highlights

- Built a CNN-based facial emotion recognition pipeline for 7-class image classification.
- Implemented an end-to-end workflow covering preprocessing, model training, validation, and evaluation.
- Used TensorFlow/Keras with checkpointing and metric tracking to improve reproducibility.
- Structured the repository for maintainability, portability, and GitHub presentation.

## Possible Improvements

- add data augmentation for generalization
- add early stopping and learning-rate scheduling
- expose inference through FastAPI or Streamlit
- containerize with Docker
- track experiments with MLflow

## Git Commands to Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit - face emotion recognition project"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

## Notes

Your original notebook-style Python script was converted into a cleaner project structure suitable for GitHub and resume sharing.
