import os
import cv2
import numpy as np
from common.feature_extractor import create_feature_extractor

def load_dataset(base_dir, img_size=(64, 64), num_frames=25):
    X, y = [], []

    feature_extractor = create_feature_extractor()

    label_map = {"v": 0, "ote": 1, "not": 2}

    for label_name, label_idx in label_map.items():
        label_dir = os.path.join(base_dir, label_name)

        if not os.path.exists(label_dir):
            print(f"Warning: {label_dir} not found, skipping.")
            continue

        for folder in os.listdir(label_dir):
            folder_path = os.path.join(label_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            frame_files = sorted(os.listdir(folder_path))
            if len(frame_files) != num_frames:
                print(f"Skipped: {folder_path} (frame count = {len(frame_files)})")
                continue

            frames = []
            for f in frame_files:
                img_path = os.path.join(folder_path, f)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read: {img_path}")
                    continue
                img = cv2.resize(img, img_size)
                img = img.astype("float32") / 255.0
                frames.append(img)

            if len(frames) != num_frames:
                continue

            frames = np.array(frames)
            
            features = feature_extractor.predict(frames)
            
            features = np.expand_dims(features, axis=-1)

            X.append(features)
            y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    print(f"\nLoaded dataset: {len(X)} samples (classes: {len(label_map)})")
    return X, y
