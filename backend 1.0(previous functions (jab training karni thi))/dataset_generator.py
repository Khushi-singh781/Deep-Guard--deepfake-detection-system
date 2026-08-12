# backend/dataset_generator.py
import os, random, glob
import numpy as np
from tensorflow.keras.utils import Sequence
import cv2

class FaceFrameGenerator(Sequence):
    """
    A Keras Sequence (data generator) to efficiently load and preprocess 
    face crop images from CSV splits for model training.
    """
    def __init__(self, split_csv, batch_size=16, img_size=(299,299), shuffle=True):
        import pandas as pd
        self.df = __import__('pandas').read_csv(split_csv)
        self.split_csv = split_csv # <--- FIX: Store the split file path
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.prepare_file_list()
        if self.shuffle: random.shuffle(self.files)

    def prepare_file_list(self):
        """Builds a list of (image_path, label) tuples from the video split CSV."""
        files=[]
        for _, r in self.df.iterrows():
            label = 1 if r['label']=='fake' else 0 # 1 for fake, 0 for real
            # Point to the face crops folder (e.g., replace 'dataset/' with 'dataset_crops/')
            frames_dir = r['frames_dir'].replace('dataset/','dataset_crops/')
            
            # Look for the cropped images (saved as *_face*.jpg)
            imgs = glob.glob(os.path.join(frames_dir,'*_face*.jpg'))
            
            for im in imgs:
                files.append((im,label))
        self.files = files
        # <--- FIX: Use self.split_csv for the print statement
        print(f"Generator loaded {len(self.files)} frames from {len(self.df)} videos for {os.path.basename(os.path.dirname(self.split_csv))}.")

    def __len__(self):
        """Returns the number of batches per epoch."""
        return max(1, len(self.files)//self.batch_size)

    def on_epoch_end(self):
        """Shuffle file list after each epoch if shuffling is enabled."""
        if self.shuffle: random.shuffle(self.files)

    def __getitem__(self, idx):
        """Generates one batch of data."""
        batch = self.files[idx*self.batch_size:(idx+1)*self.batch_size]
        X = []
        y = []
        for p,lab in batch:
            img = cv2.imread(p)
            if img is None: continue 
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            
            # Normalization (Crucial for XceptionNet: scale to 0-1)
            img = img.astype('float32')/255.0
            
            X.append(img)
            y.append(lab)
            
        return np.array(X), np.array(y)