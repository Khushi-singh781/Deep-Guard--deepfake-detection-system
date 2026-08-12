# eval_video_level.py
import numpy as np, os, glob
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
import cv2
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = 'models/xception_best.h5'
TEST_SPLIT_PATH = 'dataset/splits/test.csv'
MAX_FRAMES_PER_VIDEO = 300 # Limit frames for faster evaluation
IMG_SIZE = 299
# ---------------------

# Load Model
try:
    # Use compile=False when loading a model saved with ModelCheckpoint
    model = load_model(MODEL_PATH, compile=False)
except:
    print(f"Error: Model not found at {MODEL_PATH}. Run training first.")
    exit()

# Load Test Metadata
meta = pd.read_csv(TEST_SPLIT_PATH)
y_true = []
y_pred = []

print(f"Evaluating {len(meta)} test videos...")

for _, r in tqdm(meta.iterrows(), total=len(meta), desc="Video Evaluation"):
    frames_dir = r['frames_dir'].replace('dataset/','dataset_crops/')
    
    # Only load image paths that were created during preprocessing
    imgs = glob.glob(os.path.join(frames_dir,'*_face*.jpg')) 
    imgs = imgs[:MAX_FRAMES_PER_VIDEO] 
    
    if len(imgs)==0:
        continue
        
    X = []
    for p in imgs:
        img = cv2.imread(p)
        if img is None: continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        X.append(img)
        
    if len(X) == 0: continue
    
    # Predict on all sampled frames
    preds = model.predict(np.array(X, dtype='float32'), batch_size=32).squeeze()
    
    # Aggregate: Calculate the mean prediction across all frames
    mean_p = preds.mean()
    
    y_pred.append(mean_p)
    y_true.append(1 if r['label']=='fake' else 0)

print('\n--- Video-Level Evaluation Results ---')
print(f"Total Videos Evaluated: {len(y_true)}")
print('ROC AUC:', roc_auc_score(y_true, y_pred))

# Calculate Classification Report (using 0.5 threshold)
yhat = [1 if p>=0.5 else 0 for p in y_pred]
print(classification_report(y_true, yhat, target_names=['REAL', 'FAKE'], digits=4))