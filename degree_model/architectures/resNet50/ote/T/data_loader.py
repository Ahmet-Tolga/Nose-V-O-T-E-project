import os
import cv2
import numpy as np
from common.constants import FRAME_COUNT,IMG_SIZE
from common.feature_extractor import create_feature_extractor
from common.label_getter import get_label_t

def load_data(FOLDER_PATH,IMG_SIZE=IMG_SIZE):
    feature_extractor = create_feature_extractor()

    X = []
    y = []

    for video_folder in os.listdir(FOLDER_PATH):
        folder_full_path = os.path.join(FOLDER_PATH, video_folder)

        if not os.path.isdir(folder_full_path):
            continue

        try:
            label = int(get_label_t(video_folder))

        except:
            continue

        frame_files = sorted(os.listdir(folder_full_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
        frames = []

        for file_name in frame_files[:FRAME_COUNT]:
            img_path = os.path.join(folder_full_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  
            frames.append(img)

        if len(frames) != FRAME_COUNT:
            continue

        frames = np.array(frames)

        features = feature_extractor.predict(frames, verbose=0)
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)

