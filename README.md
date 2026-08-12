# DeepGuard 2.0 — Deepfake Detector

## How to Run
1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r backend/requirements.txt`
3. `bash run_backend.sh`
4. Open http://127.0.0.1:5001

## Features
- Face extraction via MTCNN  
- Pretrained CNN model inference  
- Flask + HTML interface  
- GPU acceleration (Apple M2 Metal)  

## Credits
Hugging Face model: `brmk/deepfake-detection-model`
