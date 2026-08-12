import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from backend.dataset_generator import FaceFrameGenerator # Assuming this path is correct

import os

# --- Configuration (Tuned for Overfitting) ---
IMG = (299, 299)
BATCH = 8
EPOCHS = 20 # Increased total epochs for long-term training capacity
LEARNING_RATE = 5e-5 
L2_REG = 1e-4 
DROPOUT_RATE = 0.6 # 🔴 INCREASED DROPOUT to fight overfitting
MODEL_SAVE_PATH = 'models/xception_best.h5' 
# ---------------------

# Load Data Generators (Ensure FaceFrameGenerator path is correct relative to this script)
train_gen = FaceFrameGenerator('dataset/splits/train.csv', batch_size=BATCH, img_size=IMG)
val_gen   = FaceFrameGenerator('dataset/splits/val.csv', batch_size=BATCH, img_size=IMG, shuffle=False)


# --- RESUME LOGIC (Checks for existing model) ---
if os.path.exists(MODEL_SAVE_PATH):
    # Resuming: If the model file exists, load it directly
    print(f"Loading existing model from: {MODEL_SAVE_PATH} to resume training.")
    # Ensure AUC custom object is passed during loading
    model = load_model(MODEL_SAVE_PATH, custom_objects={'AUC': tf.keras.metrics.AUC})

    # Re-freeze the base layers for safety after loading
    # Assuming model.layers[0] is the Xception base
    try:
        for layer in model.layers[0].layers: 
            layer.trainable = False
    except:
        print("Could not locate base model layers for re-freezing.")
else:
    # Starting Fresh: Build the model from scratch
    print("Starting new training run. Building XceptionNet model...")
    base = Xception(weights='imagenet', include_top=False, input_shape=(IMG[0], IMG[1], 3))
    x = GlobalAveragePooling2D()(base.output)

    x = Dropout(DROPOUT_RATE)(x) 
    x = Dense(256, activation='relu', kernel_regularizer=l2(L2_REG))(x)
    out = Dense(1, activation='sigmoid', kernel_regularizer=l2(L2_REG))(x)
    model = Model(base.input, out)

    # Freeze the base layers for the initial training phase
    for layer in base.layers: 
        layer.trainable = False
# --------------------


# Compile Model (Must be called even if loading to set the optimizer/metrics state)
model.compile(Adam(LEARNING_RATE), 
              loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Callbacks:
callbacks = [
    # Model Checkpoint: Saves best model based on validation AUC
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
    # Reduce LR: Halve learning rate if validation loss plateaus
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    # 🔴 Early Stopping: Wait 15 epochs for val_auc to improve before stopping
    EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True) 
]

print("Starting training or resuming from last best checkpoint with aggressive regularization...")

# Train Model
model.fit(train_gen, 
          validation_data=val_gen, 
          epochs=EPOCHS, 
          callbacks=callbacks)
