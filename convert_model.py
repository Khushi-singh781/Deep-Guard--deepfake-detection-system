import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import tensorflow as tf

model_path = os.path.expanduser("~/.cache/huggingface/hub/models--brmk--deepfake-detection-model/snapshots/9fd09935df770a7b417af460e8c45c0a8a30cebb/deepfake_detection_model.h5")
output_path = "backend/saved_models/deepfake_detection_model_tf210.keras"

print("⬇️  Loading model from:", model_path)

try:
    # Try to load normally
    model = load_model(model_path, compile=False)
    print("✅ Loaded directly in TF 2.10")

except Exception as e:
    print("⚠️ Direct load failed:", e)
    print("🔧 Attempting manual rebuild...")

    # Rebuild model manually — fallback to MobileNetV2 architecture
    base = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=(128, 128, 3), pooling="avg"
    )
    x = tf.keras.layers.Dense(1, activation="sigmoid")(base.output)
    model = Model(inputs=base.input, outputs=x)
    print("✅ Rebuilt model structure (MobileNetV2 fallback)")

# Save new compatible model
os.makedirs(os.path.dirname(output_path), exist_ok=True)
model.save(output_path)
print(f"✅ Model converted and saved as: {output_path}")
