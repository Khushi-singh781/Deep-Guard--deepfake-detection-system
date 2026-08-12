# Save this as backend/test_model.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import AUC 

# Import your custom data generator 
from backend.dataset_generator import FaceFrameGenerator 

# --- CONFIGURATION ---
BATCH_SIZE = 32 
MODEL_PATH = 'models/xception_best.h5' 
# 🛑 CRITICAL: Ensure this points to the CSV file for your NEW, UNSEEN test data.
TEST_CSV_PATH = 'dataset/splits/test.csv' 
# --- Configuration must match training setup ---
IMG_SIZE = (299, 299) 
# ---------------------

# --- 1. Load the Saved Model ---
print(f"Loading model from: {MODEL_PATH}")
# Load model, explicitly mapping the custom AUC class
model = load_model(MODEL_PATH, custom_objects={'AUC': AUC}) 

# 🟢 CRITICAL FIX: Re-compile the model to correctly register metrics names ('loss', 'accuracy', 'auc')
# This resolves the "TypeError: unsupported format string passed to NoneType" error when results are displayed.
model.compile(
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
print("Model recompiled with correct metrics.")


# --- 2. Prepare the Test Data Generator ---
print("Preparing Test Data Generator...")
# Pass the CSV file path as the first positional argument
test_generator = FaceFrameGenerator(
    TEST_CSV_PATH,  
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE, 
    shuffle=False, 
)

# If the generator has a '__len__' method, you can calculate steps:
steps = len(test_generator)
print(f"Generator created successfully and is ready for evaluation ({steps} steps).")


# --- 3. Evaluate the Model ---
print("\nStarting model evaluation on the test set...")
# Removed 'workers' and 'use_multiprocessing' to resolve the 'ValueError'
evaluation_results = model.evaluate(
    test_generator,
    verbose=1,
)

# --- 4. Display Results ---
metrics = dict(zip(model.metrics_names, evaluation_results))
print("\n--- TEST RESULTS ---")

# 🟢 FINAL FIX: Use .get('key', default_value) to prevent formatting 'None' if metric name is missing.
print(f"Test Loss: {metrics.get('loss', 0.0):.4f}")
print(f"Test Accuracy: {metrics.get('accuracy', 0.0):.4f}")
print(f"Test AUC: {metrics.get('auc', 0.0):.4f}")
print("--------------------")

# Use a default value of 0.0 for accuracy check as well
if metrics.get('accuracy', 0.0) < 0.85:
    print("⚠️ WARNING: Test Accuracy is below the expected 85%. This confirms OVERFITTING or data leakage in your train/val split.")
    print("Action: Review your data split CSV files and consider training longer with regularization.")
else:
    print("✅ SUCCESS: Test Accuracy is strong. Your model generalizes well!")
