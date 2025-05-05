# %% import libraries
import os
import json
import argparse
import whisper

# %% define functions
def segment(result):
    segments = []
    current = {"text": "", "start": None, "end": None}
    for seg in result["segments"]:
        if current["start"] is None:
            current["start"] = seg["start"]
        current["end"] = seg["end"]
        current["text"] += seg["text"] + " "
        if seg["text"].endswith('.'):
            segments.append(current.copy())
            current = {"text": "", "start": None, "end": None}
    if current["text"]:
        segments.append(current)    
    with open(f"{args.output_file}_segments.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)   
    return 


# %% parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')

args = parser.parse_args()

# %% convert video to audio
if not os.path.isfile(f'{args.output_file}_audio.wav'):
    os.system(f'ffmpeg -i {args.input_file}.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le -n {args.output_file}_audio.wav')
else:
    print(f'Audio already extracted. Step skipped.')

# %% transcribe and segment audio
if (not os.path.isfile(f'{args.output_file}_transcript.txt')) or (not os.path.isfile(f'{args.output_file}_segments.json')):
    model = whisper.load_model('small.en')
    result = model.transcribe(f'{args.output_file}_audio.wav')
    transcript = result['text']
    with open(f'{args.output_file}_transcript.txt', 'w') as f:
        f.write(transcript)
    segment(result)
else:
    print(f'Audio already transcribed and segmented. Step skipped.')
