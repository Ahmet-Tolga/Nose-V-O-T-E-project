import os
import subprocess

input_folder = './newFolder'
output_folder = './fix_output'
arr = [i for i in range(251,278)]

os.makedirs(output_folder, exist_ok=True)

all_videos = [f for f in os.listdir(input_folder) if f.endswith('.mov')]

for video in all_videos:
    video_index_str = video.split('_')[0]
    
    try:
        video_index = int(video_index_str)
    except ValueError:
        print(f"Skipped: {video} does not start with a valid number.")
        continue

    if video_index in arr:
        input_path = os.path.join(input_folder, video)
        output_path = os.path.join(output_folder, video)

        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-crf', '23',
            '-preset', 'fast',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            output_path
        ]

        print(f"Processing {video}...")
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Done: {video}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video}: {e}")
