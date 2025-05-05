# %% import libraries
import subprocess
import argparse
import os

# %% parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--video_link')
parser.add_argument('--file_name')
args = parser.parse_args()
# video_link = 'https://www.youtube.com/watch?v=dARr3lGKwk8'
# file_name = 'lecture'
path = f'files/{args.file_name}'

# %% set up directories
os.makedirs('files', exist_ok=True)

# %% download video
print('### Downloading video ###')
subprocess.run(['python', 'scripts/download_video.py',
                '--video_link', args.video_link,
                '--output_file', path])
print('### Downloaded Video ###')

# %% extract audio, transcribe audio, and segment transcript
print('\n### Transcribing audio and segmenting transcript ###')
subprocess.run(['python', 'scripts/transcribe.py',
                '--input_file', path,
                '--output_file', path])
print('### Transcribed audio and segmented transcript ###')

# %% extract frames at scene change
print('\n### Extracting frames ###')
subprocess.run(['python', 'scripts/extract_frames.py',
                '--input_file', path,
                '--frames_path', path+'_frames/'])
print('### Frames extracted ###')

# %% embed text
print('\n### Embedding chunks ###')
subprocess.run(['python', 'scripts/embed_text.py',
                   '--file_name', args.file_name,
                   '--input_file', path+'_segments',
                   '--output_file', path+'_text_embeddings'])
print('### Embedded chunks ###')

# %% embed frames
print('\n### Embedding frames ###')
subprocess.run(['python', 'scripts/embed_frames.py',
                   '--file_name', args.file_name,
                   '--input_path', path+'_frames/',
                   '--output_file', path+'_frames_embeddings'])
print('### Embedded frames ###')

