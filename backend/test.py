# backend/test_backend.py
import requests
import json

def test_health():
    """Test if backend is running"""
    try:
        response = requests.get("http://localhost:5001/health")
        print(f"🏥 Health check: {response.status_code}")
        print(f"📊 Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_analysis(video_path):
    """Test analysis endpoint with a video file"""
    try:
        with open(video_path, 'rb') as f:
            files = {'file': ('test_video.mp4', f, 'video/mp4')}
            response = requests.post("http://localhost:5001/analyze", files=files)
            
            print(f"🔍 Analysis test: {response.status_code}")
            result = response.json()
            print(f"📊 Result: {json.dumps(result, indent=2)}")
            return result
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        return None

if __name__ == "__main__":
    print("🧪 Testing DeepGuard Backend")
    
    # Test health endpoint
    if test_health():
        # Test analysis with a video file
        test_video = "test_video.mp4"
        import os
        if os.path.exists(test_video):
            test_analysis(test_video)
        else:
            print(f"⚠️ Test video not found: {test_video}")
            print("💡 Please provide a video file named 'test_video.mp4'")