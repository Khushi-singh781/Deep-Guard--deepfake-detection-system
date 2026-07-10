# DeepGuard 2.0

DeepGuard 2.0 is a deepfake video detection system developed using TensorFlow, Flask, OpenCV, and MTCNN. The application analyzes uploaded videos by extracting facial regions, preprocessing them, and classifying them using a fine-tuned Xception neural network.

The project provides a complete end-to-end pipeline including dataset preparation, face extraction, model training, evaluation, and deployment through a web interface.

---

## Features

- Deepfake video detection using a fine-tuned Xception model
- Face detection and extraction using MTCNN
- Video preprocessing pipeline
- Automated dataset preparation and splitting
- Model training with checkpointing and early stopping
- Video-level inference and confidence scoring
- Flask-based REST API
- Responsive HTML, CSS, and JavaScript frontend
- Docker support for deployment
- Evaluation scripts for testing model performance

---

## Technology Stack

### Backend

- Python
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- MTCNN
- Scikit-learn

### Frontend

- HTML
- CSS
- JavaScript

### Model

- Xception (Transfer Learning)
- Binary Classification

---

## Project Structure

```
DeepGuard-2.0/

├── backend/
│   ├── app.py
│   ├── infer_helper.py
│   ├── dataset_generator.py
│   ├── preprocess_faces.py
│
├── dataset/
│   ├── real/
│   ├── fake/
│   ├── metadata.csv
│   └── splits/
│
├── dataset_crops/
│
├── models/
│   └── xception_best.h5
│
├── frontend/
│   └── deepguard.html
│
├── prepare_metadata.py
├── split_videos.py
├── train_xception.py
├── test_model.py
├── eval_models.py
├── Dockerfile
└── README.md
```

---

## System Workflow

```
Video Upload
      │
      ▼
Frame Extraction
      │
      ▼
Face Detection (MTCNN)
      │
      ▼
Face Cropping
      │
      ▼
Image Preprocessing
      │
      ▼
Xception Model
      │
      ▼
Frame Predictions
      │
      ▼
Video-level Aggregation
      │
      ▼
Real / Fake Classification
```

---

## Dataset Preparation

Organize the dataset in the following format:

```
dataset/

├── real/
│   ├── video_001/
│   ├── video_002/
│
└── fake/
    ├── video_001/
    ├── video_002/
```

Each video folder should contain extracted image frames.

---

## Installation

Clone the repository.

```bash
git clone https://github.com/yourusername/DeepGuard-2.0.git

cd DeepGuard-2.0
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate the virtual environment.

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install the required packages.

```bash
pip install -r requirements.txt
```

---

## Preparing the Dataset

### Generate Metadata

```bash
python prepare_metadata.py
```

This creates

```
dataset/metadata.csv
```

### Split the Dataset

```bash
python split_videos.py
```

This generates

```
dataset/splits/

train.csv
val.csv
test.csv
```

### Extract Face Crops

```bash
python preprocess_faces.py
```

Face images are saved inside

```
dataset_crops/
```

---

## Training

Start model training.

```bash
python train_xception.py
```

The training pipeline includes:

- Transfer Learning using Xception
- Model Checkpointing
- Early Stopping
- Reduce Learning Rate on Plateau
- L2 Regularization
- Dropout

The best-performing model is saved as

```
models/xception_best.h5
```

---

## Model Evaluation

Evaluate the trained model.

```bash
python test_model.py
```

or

```bash
python eval_models.py
```

Evaluation metrics include:

- Accuracy
- Loss
- ROC-AUC
- Precision
- Recall
- F1 Score
- Classification Report

---

## Running the Application

Start the Flask server.

```bash
python backend/app.py
```

Open your browser and navigate to

```
http://localhost:5001
```

Upload a video and start the analysis.

---

## REST API

### Analyze Video

```
POST /api/analyze
```

Example response

```json
{
    "score": 0.91,
    "faces": 42,
    "prediction": "FAKE"
}
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | Xception |
| Input Size | 299 × 299 |
| Optimizer | Adam |
| Learning Rate | 5e-5 |
| Batch Size | 8 |
| Loss Function | Binary Crossentropy |
| Output Activation | Sigmoid |
| Regularization | L2 |
| Dropout | 0.6 |

---

## Inference Pipeline

```
Input Video
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
Image Resize (299 × 299)
      │
      ▼
Normalization
      │
      ▼
Model Prediction
      │
      ▼
Probability Aggregation
      │
      ▼
Final Classification
```

---

## Future Work

- Vision Transformer-based models
- Temporal feature learning
- Explainable AI using Grad-CAM
- Live webcam detection
- Mobile application
- Cloud deployment
- Batch video processing

---

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new feature branch.

```bash
git checkout -b feature-name
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push to your branch.

```bash
git push origin feature-name
```

5. Open a Pull Request.

---

## License

This project is licensed under the MIT License.

---

## Author

Developed by

- Khushi Singh

Department of Computer Science and Engineering

Cyber Security and Forensics

MIT World Peace University

---

## Acknowledgements

This project makes use of several open-source libraries and frameworks, including TensorFlow, OpenCV, Flask, MTCNN, NumPy, Pandas, and Scikit-learn. Their contributions have made the development of this project possible.
