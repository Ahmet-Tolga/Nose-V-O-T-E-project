import os
import cv2
import numpy as np
from common.constants import IMG_SIZE,FRAME_COUNT

def load_dataset(base_dir, img_size=(IMG_SIZE,IMG_SIZE), num_frames=FRAME_COUNT):
    data, labels = [], []
    
    class_names = ["v", "ote", "not"]
    
    for label_idx, label_name in enumerate(class_names):
        label_dir = os.path.join(base_dir, label_name)
        if not os.path.exists(label_dir):
            print(f"{label_dir} can not be found, skipping.")
            continue
        
        for folder in os.listdir(label_dir):
            folder_path = os.path.join(label_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            
            frame_files = sorted(os.listdir(folder_path))
            
            if len(frame_files) != num_frames:
                print(f"{folder_path} (number of frames = {len(frame_files)})")
                continue
            
            frames = []
            for f in frame_files:
                img_path = os.path.join(folder_path, f)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"{img_path} can not be read, skipping video.")
                    continue
                img = cv2.resize(img, img_size)
                img = np.expand_dims(img, axis=-1)
                frames.append(img)
            
            frames = np.array(frames)
            data.append(frames)
            labels.append(label_idx)
    
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels)
    
    data /= 255.0
    
    print(f"Total number of videos: {len(data)}")
    print(f"data shape: {data.shape}, label shape: {labels.shape}")
    
    return data, labels
