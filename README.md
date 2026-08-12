# DeepGuard

An end-to-end **Deepfake Video Detection System** built using **TensorFlow, XceptionNet, MTCNN, OpenCV, and Flask**. DeepGuard analyzes uploaded videos by extracting facial regions, processing them through a deep learning model, and classifying them as **Authentic** or **Deepfake**.

The project covers the complete machine learning workflow, including dataset preprocessing, metadata generation, model training, evaluation, inference, and deployment through a web-based interface.

---

## Author

**Khushi Singh**

---

## Table of Contents

- Overview
- Features
- System Architecture
- Technology Stack
- Project Structure
- Installation
- Usage
- Model Pipeline
- Results
- Future Scope
- License

---

# Overview

DeepGuard is a binary deepfake detection system designed to identify manipulated facial videos using deep learning and computer vision techniques.

Instead of processing every frame in a video, the system intelligently samples representative frames, detects faces using MTCNN, preprocesses them, and classifies them using a fine-tuned Xception neural network. Individual frame predictions are aggregated to generate a final prediction for the entire video.

The project has been developed with a modular architecture so that each stage of the machine learning pipeline can be modified, tested, or extended independently.

---

# Features

- Deepfake video detection using XceptionNet
- Face detection with MTCNN
- Automatic frame sampling
- Face preprocessing and normalization
- Video-level prediction aggregation
- Flask REST API backend
- Browser-based user interface
- Transfer learning implementation
- Dataset preprocessing utilities
- Metadata generation
- Automatic dataset splitting
- Model evaluation pipeline
- Modular project architecture

---

# System Architecture

```text
                 Uploaded Video
                        │
                        ▼
               Representative Frames
                        │
                        ▼
                Face Detection (MTCNN)
                        │
                        ▼
                 Face Preprocessing
                        │
                        ▼
              Xception Neural Network
                        │
                        ▼
             Frame-Level Predictions
                        │
                        ▼
            Video-Level Aggregation
                        │
                        ▼
          Authentic / Deepfake Verdict
```

---

# Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV, MTCNN |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |

---

# Project Structure

```text
DeepGuard/
│
├── app.py
├── train_xception.py
├── preprocess_faces.py
├── prepare_metadata.py
├── split_videos.py
├── dataset_generator.py
├── eval_models.py
├── infer_helper.py
├── models/
├── dataset/
├── frontend/
├── requirements.txt
└── README.md
```

---

# Installation

### Clone the repository

```bash
git clone https://github.com/<your-github-username>/DeepGuard.git

cd DeepGuard
```

### Create a virtual environment

#### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

Organize the dataset using the following directory structure.

```text
dataset/
│
├── real/
└── fake/
```

Generate metadata.

```bash
python prepare_metadata.py
```

Split the dataset into training, validation, and testing sets.

```bash
python split_videos.py
```

---

# Training

Train the model using:

```bash
python train_xception.py
```

The best-performing model checkpoint will automatically be saved inside the `models/` directory.

---

# Evaluation

Evaluate the trained model.

```bash
python eval_models.py
```

The evaluation pipeline reports:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Validation Loss

---

# Running the Application

Start the Flask backend.

```bash
python app.py
```

Open the frontend in your browser and upload a video for analysis.

---

# Model Pipeline

```text
Raw Video
     │
     ▼
Frame Sampling
     │
     ▼
Face Detection
     │
     ▼
Face Cropping
     │
     ▼
Image Normalization
     │
     ▼
XceptionNet Prediction
     │
     ▼
Frame Aggregation
     │
     ▼
Final Prediction
```

---

# Core Components

## Face Detection

DeepGuard uses **MTCNN** to detect and crop facial regions from sampled video frames before they are processed by the neural network.

## Deep Learning Model

The classification model is based on **XceptionNet**, utilizing transfer learning to distinguish between authentic and manipulated facial images.

## Video-Level Classification

Rather than relying on a single frame, DeepGuard combines predictions from multiple sampled frames to produce a stable and reliable prediction for the complete video.

---

# Sample Output

```json
{
    "verdict": "deepfake",
    "confidence": 0.94,
    "frames_analyzed": 32,
    "processing_time": 2.81
}
```

---

# Results

The trained model is evaluated using an independent test dataset. Standard binary classification metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC are used to assess performance.

Actual performance depends on the dataset, preprocessing quality, and training configuration.

---

# Future Scope

Possible future improvements include:

- Vision Transformer (ViT) integration
- Swin Transformer implementation
- Real-time webcam detection
- Mobile application
- Docker deployment
- Cloud deployment
- Explainable AI using Grad-CAM
- Multi-face analysis
- Audio forgery detection
- Temporal deepfake detection using Transformers

---

# License

This project is intended for educational and research purposes.

---

# Acknowledgements

This project makes use of several open-source libraries, including TensorFlow, Keras, OpenCV, MTCNN, NumPy, Pandas, and Scikit-learn. Their contributions to the machine learning community have made this work possible.

---

# Author

**Khushi Singh**

Artificial Intelligence and Machine Learning Enthusiast

