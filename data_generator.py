import os
import cv2
import numpy as np

v_folder="img/v"
ote_folder="img/ote"

os.makedirs(v_folder,exist_ok=True)
os.makedirs(ote_folder,exist_ok=True)


def parse_video_name(file_name, is_v):
    variables = file_name.split("_")
    video_name = variables[0]

    if is_v:
        try:
            start_time = variables[1]
            end_time = variables[2]
            label = variables[3][0]

            start_minute = int(start_time[:2])
            start_second = int(start_time[2:])
            end_minute = int(end_time[:2])
            end_second = int(end_time[2:])

            start_time = start_minute * 60 + start_second
            end_time = end_minute * 60 + end_second

            return video_name, start_time, end_time, label
        except Exception as e:
            print(f"Error with file name (v): {file_name} => {e}")
            return None

    else:
        try:

            if(len(variables)<5):
                return
            start_time = variables[4]
            end_time = variables[5]
            label = str(variables[6][0]) + str(variables[7][0]) + variables[8][0]

            start_minute = int(start_time[:2])
            start_second = int(start_time[2:])
            end_minute = int(end_time[:2])
            end_second = int(end_time[2:])

            start_time = start_minute * 60 + start_second
            end_time = end_minute * 60 + end_second

            return video_name, start_time, end_time, label
        except Exception as e:
            print(f"Error with file name (ote): {file_name} => {e}")
            return None



def get_frame(video_path,is_v):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: {video_path} is not opened!")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Error: can not be read FPS for {video_path}.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = total_frames / fps

    file_name = os.path.basename(video_path).split(".")[0]

    parsed = parse_video_name(file_name, is_v)
    if not parsed:
        return None

    video_name, start_time, end_time, label = parsed

    if end_time > video_length:
        print(f"Error: in {file_name}  End time ({end_time}) that is bigger than ({video_length:.2f}).")
        return

    start_frame = int((start_time) * fps)
    end_frame = int((end_time) * fps)

    if start_frame >= total_frames or end_frame >= total_frames:
        print(f"Error: {file_name} determined interval ({start_frame}-{end_frame}), Total frame count ({total_frames}) is too high!.")
        return

    frame_indices = np.linspace(start_frame, end_frame, num=25, dtype=int)

    return cap,frame_indices,video_name,total_frames,label


def enhance_frame(frame,crop_x=100, crop_y = 30,crop_w=400, crop_h = 350 ):

    cropped_frame=frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

    gray=cv2.cvtColor(cropped_frame,cv2.COLOR_BGR2GRAY)

    _,threshold=cv2.threshold(gray,25,120,cv2.THRESH_BINARY)

    kernel = np.array([[ -1, -1, -1],
                        [ -1,  9, -1],
                        [ -1, -1, -1]])


    sharpened = cv2.filter2D(gray, -1, kernel)

    return sharpened

def get_frame_for_v(video_path, output_dir):
    cap,frame_indices,video_name,total_frames,label=get_frame(video_path,True)

    video_name=f"{video_name}_{label}"

    output_dir = f"{output_dir}/{video_name}"

    if os.path.exists(output_dir):
        print(f"Skipped: {output_dir} already exists.")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0

    print(frame_indices)
    for frame_num in frame_indices:

        if frame_num >= total_frames:
            print(f"Warning: For {video_name} ({frame_num}), Total frame count ({total_frames}) is too high!.")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        cap.grab()
        ret, frame = cap.read()

        if not ret:
            print(f"Error: For {video_name} {frame_num} frame can not be read!.")
            continue

        frame_filename = f"{label}_{frame_num}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)

        enhanced_frame=enhance_frame(frame)
        cv2.imwrite(frame_path, enhanced_frame)
        
        frame_count += 1

    print(f"For {video_name} ,{frame_count} Image is saved!.")
    cap.release()
    cv2.destroyAllWindows()


def get_frame_for_ote(video_path, output_dir):
    result = get_frame(video_path, False)

    if result is None:
        print(f"Skipped: {video_path} is not a valid ote video.")
        return

    cap, frame_indices, video_name, total_frames, label = result

    video_name=f"{video_name}_{label}"

    output_dir = f"{output_dir}/{video_name}"

    if os.path.exists(output_dir):
        print(f"Skipped: {output_dir} already exists.")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0

    print(frame_indices)
    for frame_num in frame_indices:

        if frame_num >= total_frames:
            print(f"Warning: For {video_name} ({frame_num}), Total frame count ({total_frames}) is too high!.")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        cap.grab()
        ret, frame = cap.read()

        if not ret:
            print(f"Error: For {video_name} {frame_num} frame can not be read!.")
            continue

        frame_filename = f"{label}_{frame_num}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)

        enhanced_frame=enhance_frame(frame)
        cv2.imwrite(frame_path, enhanced_frame)
        
        frame_count += 1

    print(f"For {video_name} ,{frame_count} Image is saved!.")
    cap.release()
    cv2.destroyAllWindows()


for video in os.listdir("./data"):
    video_path=os.path.join("./data",video)
    get_frame_for_v(video_path,v_folder)

for video in os.listdir("./data"):
    video_path=os.path.join("./data",video)

    get_frame_for_ote(video_path,ote_folder)





