# backend/app.py
import os
import time
import sys
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
from mtcnn import MTCNN
import tensorflow as tf

# ============================================================
# 🧩 Set deterministic behaviour for consistent predictions
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 🔧 Configuration
# ============================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "frontend_uploads")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_FRAMES_TO_CHECK = 32     # more frames = stable results
FRAME_STRIDE = 10

# ============================================================
# 🚀 Flask app setup
# ============================================================
app = Flask(__name__, static_folder=FRONTEND_DIR, template_folder=FRONTEND_DIR)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024 * 1024  # 1 GB

# ============================================================
# 🧠 Load the DeepGuard model (TF 2.10 converted version)
# ============================================================
MODEL_FILENAME = "deepfake_detection_model_tf210.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

print("🧠 Loading local TF 2.10-compatible DeepGuard model...")
print(f"🔍 Model path: {MODEL_PATH}")
print(f"📁 Model exists: {os.path.exists(MODEL_PATH)}")
print(f"📂 Model directory contents: {os.listdir(MODEL_DIR)}")

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✅ Model loaded successfully from: {MODEL_PATH}")
        # Print model input shape for debugging
        print(f"📐 Model input shape: {model.input_shape}")
    else:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
except Exception as e:
    print("⚠️ Failed to load model:", e)
    print("⚙️ Using MobileNetV2 fallback for demo mode.")
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    # Updated to 128x128 to match the main model
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=x)
    print("✅ Fallback model created successfully")
    print(f"📐 Fallback model input shape: {model.input_shape}")

print("✅ Model ready:", model.name)

# ============================================================
# 👁️ Initialize MTCNN face detector
# ============================================================
try:
    detector = MTCNN()
    print("✅ MTCNN face detector initialized")
except Exception as e:
    print(f"❌ MTCNN initialization failed: {e}")
    detector = None

# ============================================================
# 🧩 Helper Functions
# ============================================================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def extract_faces_from_video(video_path, max_frames=MAX_FRAMES_TO_CHECK):
    """
    Extract evenly spaced face frames from the video deterministically.
    Returns np.array of faces normalized to 0..1.
    """
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
    
    # ✅ choose evenly spaced frames for consistent analysis
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    print(f"🎯 Analyzing {len(frame_indices)} frames: {frame_indices}")

    faces_found = 0
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Could not read frame {frame_idx}")
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            dets = detector.detect_faces(rgb)
        except Exception as e:
            print(f"⚠️ Face detection failed on frame {frame_idx}: {e}")
            dets = []

        if dets:
            # pick largest detected face
            det = max(dets, key=lambda d: d["box"][2] * d["box"][3])
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
        else:
            print(f"⚠️ No faces detected in frame {frame_idx}")

    cap.release()
    print(f"📸 Extracted {faces_found} faces from {len(frame_indices)} frames")

    if len(faces) == 0:
        return np.zeros((0, 128, 128, 3), dtype=np.float32)  # Updated to 128

    return np.stack(faces, axis=0)


def analyze_video_file(video_path):
    """
    Run deterministic analysis on uploaded video.
    Returns a stable JSON result.
    """
    t0 = time.time()
    try:
        print(f"🔍 Analyzing video: {video_path}")
        
        # Extract faces
        faces = extract_faces_from_video(video_path)
        print(f"📸 Extracted {faces.shape[0]} faces")
        
        if faces.shape[0] == 0:
            return {
                "verdict": "no_faces_detected",
                "confidence": 0.0,
                "frames_analyzed": 0,
                "processing_time": round(time.time() - t0, 2)
            }

        # Make predictions
        print("🧠 Running model predictions...")
        print(f"📐 Input shape to model: {faces.shape}")
        preds = model.predict(faces, verbose=1).reshape(-1)  # Set verbose=1 to see progress
        print(f"📊 Predictions: {preds}")
        
        avg_prob = float(np.median(preds))   # ✅ median = stable result
        verdict = "fake" if avg_prob >= 0.5 else "real"

        result = {
            "verdict": verdict,
            "confidence": round(avg_prob, 4),
            "frames_analyzed": int(faces.shape[0]),
            "processing_time": round(time.time() - t0, 2)
        }
        
        print(f"✅ Analysis complete: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": "internal analysis error", 
            "detail": str(e)
        }

# ============================================================
# 🌐 Flask Routes
# ============================================================
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "deepguard.html")


@app.route("/<path:path>")
def serve_frontend_asset(path):
    target = os.path.join(FRONTEND_DIR, path)
    if not os.path.exists(target):
        abort(404)
    return send_from_directory(FRONTEND_DIR, path)


@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "file type not allowed"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    print(f"📥 Received file: {filename}")
    
    try:
        file.save(save_path)
        print(f"✅ File saved: {save_path} (size: {os.path.getsize(save_path)} bytes)")
        
        result = analyze_video_file(save_path)
        print(f"📊 Final result: {result}")
        
    except Exception as e:
        print(f"❌ Endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        result = {"error": "internal analysis error", "detail": str(e)}
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(save_path):
                os.remove(save_path)
                print(f"🗑️ Cleaned up: {save_path}")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    return jsonify(result)


@app.route("/health")
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "detector_ready": detector is not None,
        "model_name": model.name if model else "fallback",
        "model_input_shape": str(model.input_shape) if model else "unknown"
    }
    return jsonify(status)


# ============================================================
# 🏁 Run server
# ============================================================
if __name__ == "__main__":
    port = 5001
    for arg in sys.argv[1:]:
        if arg.startswith("--port="):
            try:
                port = int(arg.split("=", 1)[1])
            except:
                pass
    print(f"🚀 Starting DeepGuard backend on port {port}")
    print(f"📁 Frontend directory: {FRONTEND_DIR}")
    print(f"💾 Upload directory: {UPLOAD_FOLDER}")
    app.run(host="0.0.0.0", port=port, debug=True)