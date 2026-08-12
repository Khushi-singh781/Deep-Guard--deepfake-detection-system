# backend/infer_helper.py
import tensorflow as tf, numpy as np, cv2, os
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Global caches for model and detector
MODEL=None
DETECTOR=None
IMG_SIZE = 299 # Xception input size

def get_model():
    """Loads and caches the Keras model."""
    global MODEL
    if MODEL is None:
        try:
            MODEL = load_model('models/xception_best.h5', compile=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError("Deepfake model failed to load. Ensure 'models/xception_best.h5' exists.")
    return MODEL

def get_detector():
    """Loads and caches the MTCNN face detector."""
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = MTCNN()
    return DETECTOR

def video_to_score(video_path, sample_frames=64):
    """
    Analyzes a video file for deepfake classification.
    Steps: Frame Sampling -> Face Detection -> Frame-level Prediction -> Aggregation
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'Could not open video file.'}
        
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total-1), min(sample_frames, total)).astype(int)
    frames=[]
    
    # 1. Sample Frames
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
    cap.release()
    
    if len(frames) == 0:
        return {'error': 'No frames could be read from the video.'}

    # 2. Face Detection and Cropping
    det = get_detector()
    crops=[]
    
    for f in frames:
        res = det.detect_faces(f)
        for r in res:
            x,y,w,h = r['box']; 
            x,y = max(0,x), max(0,y)
            # Clamp coordinates
            c = f[y:min(y+h, f.shape[0]), x:min(x+w, f.shape[1])]
            
            if c.size==0: continue
            
            # Resize and normalize
            c = cv2.resize(c, (IMG_SIZE, IMG_SIZE)) / 255.0
            crops.append(c)
            
    if len(crops)==0:
        return {'error':'No faces detected in sampled frames. Try a video with visible faces.'}
        
    # 3. Model Prediction and Aggregation
    model = get_model()
    preds = model.predict(np.array(crops, dtype='float32'), batch_size=16).squeeze()
    
    # Calculate mean score across all detected face crops
    if preds.ndim == 0:
        mean_p = float(preds)
    else:
        mean_p = float(preds.mean())
        
    return {'score':mean_p, 'faces':len(crops)}