import os
import cv2
import numpy as np
from common.constants import FRAME_COUNT, IMG_SIZE
from common.label_getter import get_label_o

def load_videos(folder, frame_count=FRAME_COUNT, img_size=IMG_SIZE):
    videos = []
    labels = []

    for video_folder in os.listdir(folder):
        video_path = os.path.join(folder, video_folder)
        if os.path.isdir(video_path):
            try:
                label=int(get_label_o(video_folder))
            except ValueError:
                print(f"[SKIP] invalid dir name: {video_folder}")
                continue

            frames = []
            frame_files = sorted(os.listdir(video_path))[:frame_count]
            print(f"[INFO] {video_folder} -> {len(frame_files)} frame")

            for filename in frame_files:
                img_path = os.path.join(video_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                    img = img.flatten()
                    frames.append(img)
                else:
                    print(f"[WARN] frame can not be read!: {img_path}")

            if len(frames) == frame_count:
                videos.append(frames) 
                labels.append(label)
            else:
                print(f"[SKIP] {video_folder} -> not enough frame ({len(frames)}/{frame_count})")

    print(f"[SUMMARY] Sum of videos: {len(videos)}")
    return np.array(videos), np.array(labels)
