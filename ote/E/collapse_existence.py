import os
import cv2
import numpy as np
import random as rn
from common.constants import IMAGE_SIZE, FRAME_COUNT , VIDEO_NUM
from common.feature_extractor import create_feature_extractor
from common.label_getter import get_label_e

feature_extractor = create_feature_extractor()

def load_data_for_collapse_existence(folder_path):

    X = []
    y = []

    for video_folder in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, video_folder)

        if not os.path.isdir(folder_full_path):
            continue

        try:
            label = int( "0" if get_label_e(video_folder)=="0" else "1")
        except:
            continue

        frame_files = sorted(os.listdir(folder_full_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
        frames = []

        for file_name in frame_files[:FRAME_COUNT]:
            img_path = os.path.join(folder_full_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img / 255.0  
            frames.append(img)

        if len(frames) != FRAME_COUNT:
            continue

        frames = np.array(frames)

        features = feature_extractor.predict(frames, verbose=0)
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def load_data_for_collapse_type(folder_path):

    X = []
    y = []

    for video_folder in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, video_folder)

        if not os.path.isdir(folder_full_path):
            continue

        try:
            label_str = get_label_e(video_folder)

            if(label_str not in ["1","2"]):
                continue
            label = int(label_str)

        except:
            continue

        frame_files = sorted(os.listdir(folder_full_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
        frames = []

        for file_name in frame_files[:FRAME_COUNT]:
            img_path = os.path.join(folder_full_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img / 255.0  
            frames.append(img)

        if len(frames) != FRAME_COUNT:
            continue

        frames = np.array(frames)

        features = feature_extractor.predict(frames, verbose=0)
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def load_data_for_final_test(folder_path):

    X = []
    y = []

    sample_count = int(40 * VIDEO_NUM / 100)
    video_names = set(rn.sample(range(1, VIDEO_NUM+1), sample_count))

    for video_folder in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, video_folder)

        if not os.path.isdir(folder_full_path):
            continue

        try:
            if(int(video_folder.split("_")[0]) not in video_names):
                continue

            label_str = get_label_e(video_folder)

            label = int(label_str)

        except:
            continue

        frame_files = sorted(os.listdir(folder_full_path), key=lambda x: int(x.split('_')[1].split('.')[0]))
        frames = []

        for file_name in frame_files[:FRAME_COUNT]:
            img_path = os.path.join(folder_full_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img / 255.0  
            frames.append(img)

        if len(frames) != FRAME_COUNT:
            continue

        frames = np.array(frames)

        features = feature_extractor.predict(frames, verbose=0)
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)

