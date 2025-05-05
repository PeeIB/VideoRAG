# %% import libraries
import os
import argparse

# %% parse argumets
parser = argparse.ArgumentParser()
parser.add_argument('--video_link')
parser.add_argument('--output_file')
args = parser.parse_args()

# %% download video
if not os.path.isfile(f'{args.output_file}.mp4'):
    os.system(f'yt-dlp --no-overwrites -o {args.output_file}.mp4 "{args.video_link}"')
else:
    print('Video already downloaded. Step skipped.')