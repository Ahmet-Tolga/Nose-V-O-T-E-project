import os

test_files=['80', '42', '307', '275', '129', '242', '283', '98', 
'0179', '213', '0195', '52', '110', '325', '304', '89', '140', 
'0176', '36', '13', '248', '253', '0189', '40', '21', '257', '14', '130', 
'151', '0185', '0201', '305', '59', '19', '0204', '295', '135', '7', '34', '0180', 
'82', '11', '218', '137', '0196', '265', '63', '0188', '147', '249', '78', '278', '212']

test_files=[int(t) for t in test_files]


def return_test_videos(video_dir):
    all_videos = [f for f in os.listdir(video_dir) if f.endswith(".mov")]

    selected_videos = []

    for f in all_videos:
        try:
            prefix = int(f.split("_")[0])
            if prefix in test_files:
                selected_videos.append(f)
        except ValueError:
            continue

    return selected_videos

