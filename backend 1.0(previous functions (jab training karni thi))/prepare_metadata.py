# prepare_metadata.py
import os, csv

ROOT = 'dataset'
OUT = 'dataset/metadata.csv'

rows = []
for label in ['real', 'fake']:
    p = os.path.join(ROOT, label)
    if not os.path.isdir(p):
        continue
    for vid in sorted(os.listdir(p)):
        vid_path = os.path.join(p, vid)
        if not os.path.isdir(vid_path):
            continue
        rows.append([vid, label, vid_path])

os.makedirs('dataset', exist_ok=True)
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['video_id', 'label', 'frames_dir'])
    w.writerows(rows)
print(f'Wrote {len(rows)} rows to {OUT}')
