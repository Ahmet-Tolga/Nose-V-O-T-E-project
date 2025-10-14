import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from common.constants import IMG_SIZE, CATEGORIES

def load_data(base_path="../../img", test_size=0.2, random_state=42):
    X_train, y_train, X_test, y_test = [], [], [], []
    test_videos = []

    video_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    train_videos, test_videos = train_test_split(video_folders, test_size=test_size, random_state=random_state)

    for video in train_videos:
        for category in CATEGORIES:
            folder = os.path.join(base_path, video, category)
            if not os.path.exists(folder):
                continue
            for filename in os.listdir(folder):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X_train.append(img)
                y_train.append(CATEGORIES.index(category))

    for video in test_videos:
        for category in CATEGORIES:
            folder = os.path.join(base_path, video, category)
            if not os.path.exists(folder):
                continue
            for filename in os.listdir(folder):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X_test.append(img)
                y_test.append(CATEGORIES.index(category))

    X_train = np.array(X_train, dtype="float32") / 255.0
    X_test = np.array(X_test, dtype="float32") / 255.0
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"Number of train video: {len(train_videos)}, Number of test video: {len(test_videos)}")
    print(f"Number of train frame: {len(X_train)}, Number of test frame: {len(X_test)}")

    return X_train, X_test, y_train, y_test, test_videos
