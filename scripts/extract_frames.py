# %% import libraries and modules
import os
import argparse
import cv2
import json
from scenedetect import detect, ContentDetector

# %% parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--frames_path')
args = parser.parse_args()

# %% make output directory
if (not os.path.isdir(args.frames_path)):
    os.makedirs(args.frames_path, exist_ok=True)
    # detect scenes
    scenes = detect(f'{args.input_file}.mp4', ContentDetector(threshold=5), show_progress=True)

    # save first frame of each scene
    cap = cv2.VideoCapture(f'{args.input_file}.mp4')
    time_stamps = []
    current = {'frameId': '', 'start': None, 'end': None}
    for i, (start_time, end_time) in enumerate(scenes):
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time.get_seconds() * 1000)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f'{args.frames_path}/scene_{i+1:02d}.jpg', frame)
        current['frameId'] = f'{i+1:02d}'
        current['start'] = start_time.get_seconds()
        current['end'] = end_time.get_seconds()
        time_stamps.append(current.copy())
    cap.release()
    with open(f"{args.frames_path}/time_stamps.json", "w", encoding="utf-8") as f:
        json.dump(time_stamps, f, ensure_ascii=False, indent=2)   

else:
    print('Frames already extracted. Step skipped.')
