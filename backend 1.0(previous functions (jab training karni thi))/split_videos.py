# split_videos.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs('dataset/splits', exist_ok=True)

meta = pd.read_csv('dataset/metadata.csv')
# 80% Train, 10% Val, 10% Test (splitting 20% into two 10% halves)
train, temp = train_test_split(meta, test_size=0.2, stratify=meta['label'], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

train.to_csv('dataset/splits/train.csv', index=False)
val.to_csv('dataset/splits/val.csv', index=False)
test.to_csv('dataset/splits/test.csv', index=False)

print('Split files written to dataset/splits/')
print(f"Train: {len(train)} videos, Val: {len(val)} videos, Test: {len(test)} videos")