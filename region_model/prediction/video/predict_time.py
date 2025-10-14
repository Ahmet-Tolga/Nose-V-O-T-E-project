import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../video/architectures/resnet50")))

from common.feature_extractor import create_feature_extractor
import cv2
import numpy as np
from collections import Counter
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(BASE_DIR)

from test_files import return_test_videos

def enhance_frame(frame, crop_x=100, crop_y=30, crop_w=400, crop_h=350):
    cropped_frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 25, 120, cv2.THRESH_BINARY)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    return sharpened


def classify_video_blocks(video_path, model_path, block_size=25, img_size=(150,150)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = load_model(model_path)
    feature_extractor = create_feature_extractor()

    block_classes = []
    frames_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_enhanced = enhance_frame(frame)
        frame_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(frame_rgb, img_size)
        img = img.astype("float32") / 255.0
        frames_buffer.append(img)

        if len(frames_buffer) == block_size:
            frames_array = np.array(frames_buffer)
            features = feature_extractor.predict(frames_array)
            features = np.expand_dims(features, axis=0)
            preds = model.predict(features)
            block_class = np.argmax(preds, axis=1)[0]
            block_classes.append(block_class)
            frames_buffer = []

    if len(frames_buffer) > 0:
        last_frame = frames_buffer[-1]
        while len(frames_buffer) < block_size:
            frames_buffer.append(last_frame)
        frames_array = np.array(frames_buffer)
        features = feature_extractor.predict(frames_array)
        features = np.expand_dims(features, axis=0)
        preds = model.predict(features)
        block_class = np.argmax(preds, axis=1)[0]
        block_classes.append(block_class)

    cap.release()
    return block_classes, fps

def get_v_ote_segments_ordered(block_classes, fps):
    def find_longest_segment(classes, target, start_index=0, end_index=None):
        if end_index is None:
            end_index = len(classes)
        max_len, max_start = 0, None
        current_len, current_start = 0, None

        for i in range(start_index, end_index):
            if classes[i] == target:
                if current_start is None:
                    current_start = i
                current_len += 1
            else:
                if current_len > max_len:
                    max_len, max_start = current_len, current_start
                current_len, current_start = 0, None

        if current_len > max_len:
            max_len, max_start = current_len, current_start

        if max_len == 0:
            return None
        return max_start, max_start + max_len - 1

    v_seg = find_longest_segment(block_classes, 0)
    if v_seg:
        v_start, v_end = v_seg
        v_start_sec = v_start * 25 / fps
        v_end_sec = (v_end + 1) * 25 / fps
    else:
        v_start_sec = v_end_sec = None
        v_end = -1

    ote_seg = find_longest_segment(block_classes, 1, start_index=(v_end + 1 if v_end >= 0 else 0))
    if ote_seg:
        ote_start, ote_end = ote_seg
        ote_start_sec = ote_start * 25 / fps
        ote_end_sec = (ote_end + 1) * 25 / fps
    else:
        ote_start_sec = ote_end_sec = None

    return v_start_sec, v_end_sec, ote_start_sec, ote_end_sec


def safe_mse(t, p):
    return None if len(t) == 0 else mean_squared_error(t, p)

def parse_filename_segments(filename):
    name = os.path.basename(filename).split(".")[0]
    parts = name.split("_")

    def to_seconds(s):
        if not s: return None
        if all(ch == '0' for ch in s):
            return None
        s = s.strip()

        try:
            if len(s) <= 2:
                return int(s)
            mins = int(s[:-2])
            secs = int(s[-2:])
            return mins * 60 + secs
        except:
            return None

    result = {"v": [], "ote": []}

    if len(parts) >= 3:
        vs = to_seconds(parts[1])
        ve = to_seconds(parts[2])
        if vs is not None and ve is not None:
            result["v"].append((vs, ve))

    if len(parts) >= 6:
        os_ = to_seconds(parts[4])
        oe = to_seconds(parts[5])
        if os_ is not None and oe is not None:
            result["ote"].append((os_, oe))

    return result

model_path = "../saved_models/resnet_bilstm_model.h5"

test_video="0195_0018_0022_2C_0057_0102_0_0_0.mov"
test_video2="242_0006_0014_2C_0028_0033_2L_0_0.mov"
test_video3="59_0028_0033_0_0118_0122_0_0_0.mov"

segments=parse_filename_segments(test_video)

print(segments)

classes, fps = classify_video_blocks(os.path.join("../../../data/",test_video3), model_path)

print([int(i) for i in classes])

v_start, v_end, ote_start, ote_end = get_v_ote_segments_ordered(classes, fps)

print(fps)
print(f"v start={v_start},end_time={v_end}")
print(f"ote start={ote_start},end_time={ote_end}")

# v_start_true, v_start_pred = [], []
# v_end_true, v_end_pred = [], []
# ote_start_true, ote_start_pred = [], []
# ote_end_true, ote_end_pred = [], []

# test_videos = return_test_videos("../../../data")

# gt_segments = {tv: parse_filename_segments(tv) for tv in test_videos}

# for test_video in test_videos:
#     video_path = os.path.join("../../../data", test_video)
#     try:
#         classes, fps = classify_video_blocks(video_path, model_path)
#         v_start, v_end, ote_start, ote_end = get_v_ote_segments_ordered(classes, fps)
#     except Exception as e:
#         print("Error While reading:", test_video, e)
#         continue

#     if test_video not in gt_segments:
#         print(f"GT can not be found {test_video} için, atlandı.")
#         continue

#     gt_v_segments = gt_segments[test_video]["v"]
#     gt_ote_segments = gt_segments[test_video]["ote"]

#     print("filename:", test_video)
#     print("  GT v:", gt_v_segments, "pred v:", (v_start, v_end))
#     print("  GT ote:", gt_ote_segments, "pred ote:", (ote_start, ote_end))

#     if gt_v_segments and v_start is not None:
#         gt_s, gt_e = gt_v_segments[0]
#         v_start_true.append(gt_s)
#         v_end_true.append(gt_e)
#         v_start_pred.append(v_start)
#         v_end_pred.append(v_end)

#     if gt_ote_segments and ote_start is not None:
#         gt_s, gt_e = gt_ote_segments[0]
#         ote_start_true.append(gt_s)
#         ote_end_true.append(gt_e)
#         ote_start_pred.append(ote_start)
#         ote_end_pred.append(ote_end)


# print("\n=== Results ===")
# print("v start MSE :", safe_mse(v_start_true, v_start_pred))
# print("v end   MSE :", safe_mse(v_end_true, v_end_pred))
# print("ote start MSE :", safe_mse(ote_start_true, ote_start_pred))
# print("ote end   MSE :", safe_mse(ote_end_true, ote_end_pred))

# # combine
# all_v_gt = v_start_true + v_end_true
# all_v_pred = v_start_pred + v_end_pred
# all_ote_gt = ote_start_true + ote_end_true
# all_ote_pred = ote_start_pred + ote_end_pred
# print("All v (start+end) MSE  :", safe_mse(all_v_gt, all_v_pred))
# print("All ote (start+end) MSE :", safe_mse(all_ote_gt, all_ote_pred))
