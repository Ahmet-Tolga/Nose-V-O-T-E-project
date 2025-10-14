import os
import cv2
import numpy as np

base_img_folder = "img"
os.makedirs(base_img_folder, exist_ok=True)

def parse_video_name(file_name):
    variables = file_name.split("_")
    video_name = variables[0]

    if len(variables) < 6:
        print(f"Error: {file_name} has too few parts.")
        return None

    try:
        v_start_time = variables[1]
        v_end_time = variables[2]
        ote_start_time = variables[4]
        ote_end_time = variables[5]

        def to_seconds(t):
            return int(t[:2]) * 60 + int(t[2:])

        v_start_time = to_seconds(v_start_time)
        v_end_time = to_seconds(v_end_time)
        ote_start_time = to_seconds(ote_start_time)
        ote_end_time = to_seconds(ote_end_time)

        return video_name, v_start_time, v_end_time, ote_start_time, ote_end_time

    except Exception as e:
        print(f"Error parsing times for {file_name}: {e}")
        return None


def enhance_frame(frame, crop_x=100, crop_y=30, crop_w=400, crop_h=350):
    cropped_frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 25, 120, cv2.THRESH_BINARY)
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    sharpened = cv2.filter2D(gray, -1, kernel)
    return sharpened


def get_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: {video_path} cannot be opened!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0:
        print(f"Error: Cannot read FPS for {video_path}.")
        return

    file_name = os.path.basename(video_path).split(".")[0]
    parsed = parse_video_name(file_name)
    if parsed is None:
        return

    video_name, v_start_time, v_end_time, ote_start_time, ote_end_time = parsed

    video_folder = os.path.join(base_img_folder, video_name)
    v_folder = os.path.join(video_folder, "v")
    ote_folder = os.path.join(video_folder, "ote")
    not_folder = os.path.join(video_folder, "not")

    os.makedirs(v_folder, exist_ok=True)
    os.makedirs(ote_folder, exist_ok=True)
    os.makedirs(not_folder, exist_ok=True)

    v_start_frame = int(v_start_time * fps)
    v_end_frame = int(v_end_time * fps)
    ote_start_frame = int(ote_start_time * fps)
    ote_end_frame = int(ote_end_time * fps)

    v_frame_indices = np.linspace(v_start_frame, v_end_frame, num=30, dtype=int)
    ote_frame_indices = np.linspace(ote_start_frame, ote_end_frame, num=30, dtype=int)
    before_v_frame_indices = np.linspace(3, v_start_frame - 1, num=10, dtype=int)
    between_v_ote_frame_indices = np.linspace(v_end_frame + 3, ote_start_frame - 3, num=10, dtype=int)
    after_ote_frame_indices = np.linspace(ote_end_frame + 3, total_frames - 1, num=10, dtype=int)

    def save_frames(indices, folder):
        for frame_num in indices:
            if frame_num >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"Error reading frame {frame_num} from {video_name}")
                continue
            enhanced_frame = enhance_frame(frame)
            frame_filename = f"{video_name}_{frame_num}.jpg"
            cv2.imwrite(os.path.join(folder, frame_filename), enhanced_frame)

    save_frames(v_frame_indices, v_folder)
    save_frames(ote_frame_indices, ote_folder)
    save_frames(before_v_frame_indices, not_folder)
    save_frames(between_v_ote_frame_indices, not_folder)
    save_frames(after_ote_frame_indices, not_folder)

    cap.release()

for video in os.listdir("../../data"):
    video_path = os.path.join("../../data", video)
    get_frame(video_path)
