# backend/preprocess_faces_optimized.py
import cv2
import os
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
IMG_SIZE = 299 # Xception input size
SRC_ROOT = 'dataset'
OUT_ROOT = 'dataset_crops'
# Use the number of available CPU cores for maximum throughput
# A good starting point is to use all available cores.
NUM_WORKERS = os.cpu_count()

# Ensure the output directory exists
os.makedirs(OUT_ROOT, exist_ok=True)

def process_video(label: str, vid: str):
    """
    Worker function to process all images within a single video directory.
    This function runs in parallel across multiple CPU cores.
    """
    # CRITICAL: MTCNN must be imported and initialized inside the worker function 
    # when using multiprocessing, as it relies on TensorFlow/Keras which is 
    # not thread-safe and needs a separate graph/session per process.
    # Importing here keeps the main script clean.
    from mtcnn import MTCNN
    detector = MTCNN()

    vid_in = os.path.join(SRC_ROOT, label, vid)
    out_vid = os.path.join(OUT_ROOT, label, vid)
    
    # Check for input directory existence (should be checked in main loop, but safe here)
    if not os.path.isdir(vid_in):
        return 0

    os.makedirs(out_vid, exist_ok=True)
    
    # Look for all common image extensions: .png, .jpg, .jpeg (case-insensitive)
    extensions = ['*.[jpJP][nN][gG]', '*.[jpJP][eE][gG]', '*.[jpJP][gG]']
    imgs = []
    for ext in extensions:
        imgs.extend(glob.glob(os.path.join(vid_in, ext)))
    
    # Sort for deterministic processing
    imgs.sort()
    
    # Simple image counter for the video (not using tqdm inside the worker)
    processed_count = 0 
    
    for imgp in imgs:
        img = cv2.imread(imgp)
        if img is None: 
            continue
                
        # 1. Color Conversion (BGR -> RGB for MTCNN)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Face Detection
        res = detector.detect_faces(rgb)
            
        for i, r in enumerate(res):
            x, y, w, h = r['box']
            
            # Basic coordinate clamping to stay in bounds
            x, y = max(0, x), max(0, y)
            
            # 3. Cropping and Clamping
            crop = rgb[y:min(y + h, rgb.shape[0]), x:min(x + w, rgb.shape[1])]
                            
            if crop.size == 0: 
                continue
            
            # 4. Resizing to XceptionNet input size
            # Note: This is a fast operation
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                
            # 5. I/O - Writing the cropped face
            out_name = os.path.basename(imgp).rsplit('.', 1)[0] + f'_face{i}.jpg'
            out_path = os.path.join(out_vid, out_name)
                            
            # Convert back to BGR for standard JPEG/OpenCV write
            cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            processed_count += 1
            
    return processed_count

def main():
    # 1. Gather all tasks (video directories)
    tasks = []
    for label in ['real', 'fake']:
        src_label = os.path.join(SRC_ROOT, label)
        if not os.path.exists(src_label): 
            continue
        
        for vid in sorted(os.listdir(src_label)):
            vid_in = os.path.join(src_label, vid)
            if os.path.isdir(vid_in):
                tasks.append((label, vid))
            
    total_tasks = len(tasks)
    total_processed_images = 0
    
    print(f"Starting parallel processing on {total_tasks} videos with {NUM_WORKERS} workers...")
    
    # 2. Use ProcessPoolExecutor to distribute the work
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        futures = [executor.submit(process_video, label, vid) for label, vid in tasks]
        
        # 3. Monitor progress using tqdm over the submitted futures
        with tqdm(total=total_tasks, desc='Overall Video Progress') as pbar:
            for future in as_completed(futures):
                try:
                    num_imgs = future.result()
                    total_processed_images += num_imgs
                except Exception as e:
                    # You might want to log the specific task that failed here
                    print(f"\nAn error occurred in a worker: {e}")
                
                pbar.update(1) # Increment the progress bar per completed video

    print(f'\nTotal images processed: {total_processed_images} (estimated 39000)')
    print(f'Face crops are ready in {OUT_ROOT}/')

if __name__ == '__main__':
    # Add an import here to ensure MTCNN/TensorFlow is not initialized in the parent process
    # which is often required to prevent issues on various OS/Python configurations.
    from mtcnn import MTCNN
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nA fatal error occurred in the main process: {e}")