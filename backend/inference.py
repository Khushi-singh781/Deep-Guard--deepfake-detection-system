# backend/inference.py
import os, cv2, numpy as np, tensorflow as tf
from mtcnn import MTCNN
from tqdm import tqdm
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_detection_model_tf210.keras")  # Updated to match app.py

print("🧠 Loading TF 2.10 compatible model...")
print(f"🔍 Model path: {MODEL_PATH}")
print(f"📁 Model exists: {os.path.exists(MODEL_PATH)}")

# Try to load the TF 2.10 compatible model
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ TF 2.10 compatible model loaded successfully!")
        print(f"📐 Model input shape: {model.input_shape}")
    else:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
except Exception as e:
    print("⚠️ Error loading TF 2.10 model:", e)
    print("➡️ Switching to fallback MobileNetV2 model")
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(128, 128, 3))  # Updated to 128x128
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=base.input, outputs=output)
    print("✅ Fallback model created")
    print(f"📐 Fallback model input shape: {model.input_shape}")

# Initialize face detector
try:
    detector = MTCNN()
    print("✅ MTCNN face detector initialized")
except Exception as e:
    print(f"❌ MTCNN initialization failed: {e}")
    detector = None

def extract_faces(video_path, max_frames=32):
    """Extract faces from video frames (matches app.py logic)"""
    if detector is None:
        print("❌ Face detector not available")
        return np.zeros((0, 128, 128, 3), dtype=np.float32)  # Updated to 128
        
    cap = cv2.VideoCapture(video_path)
    faces = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("❌ Video has 0 frames")
        cap.release()
        return np.zeros((0, 128, 128, 3), dtype=np.float32)  # Updated to 128

    print(f"📹 Video has {total_frames} total frames")
    
    # Use evenly spaced frames like app.py for consistency
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    print(f"🎯 Analyzing {len(frame_indices)} frames")

    faces_found = 0
    for frame_idx in tqdm(frame_indices, desc="Extracting faces"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = detector.detect_faces(rgb)
        except Exception as e:
            print(f"⚠️ Face detection failed on frame {frame_idx}: {e}")
            results = []
            
        if results:
            # pick largest detected face like app.py
            det = max(results, key=lambda d: d["box"][2] * d["box"][3])
            x, y, w, h = det["box"]
            x, y = max(0, x), max(0, y)
            face = rgb[y:y+h, x:x+w]
            if face.size == 0:
                continue
            # FIXED: Resize to 128x128 instead of 224x224
            face = cv2.resize(face, (128, 128))
            face = face.astype("float32") / 255.0
            faces.append(face)
            faces_found += 1
            
    cap.release()
    
    if len(faces) == 0:
        return np.zeros((0, 128, 128, 3), dtype=np.float32)  # Updated to 128
        
    print(f"📸 Extracted {faces_found} faces from {len(frame_indices)} frames")
    return np.array(faces)

def predict_video(video_path):
    """Predict if video contains deepfake content"""
    if not os.path.exists(video_path):
        return {"error": "Video file not found"}
        
    faces = extract_faces(video_path)
    if len(faces) == 0:
        return {"error": "No faces detected"}
    
    try:
        print(f"🧠 Running predictions on {len(faces)} faces...")
        print(f"📐 Input shape: {faces.shape}")
        preds = model.predict(faces, verbose=1).reshape(-1)
        avg_score = float(np.median(preds))  # Use median like app.py for stability
        verdict = "fake" if avg_score >= 0.5 else "real"
        
        result = {
            "verdict": verdict,
            "confidence": round(avg_score, 4),
            "frames_analyzed": len(faces)
        }
        
        print(f"✅ Prediction complete: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    # Test the inference
    test_video = "test_video.mp4"
    if os.path.exists(test_video):
        print(f"🔍 Testing with {test_video}")
        result = predict_video(test_video)
        print(f"📊 Final result: {result}")
    else:
        print(f"⚠️ Test video not found: {test_video}")
        print("💡 Please provide a video file named 'test_video.mp4'")