# %% import libraries and modules
import os
import json
import argparse
import numpy as np
from transformers import CLIPModel, AutoProcessor
from PIL import Image
import torch
import torch.nn.functional as F

# %% parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file_name')
parser.add_argument('--input_path')
parser.add_argument('--output_file')
args = parser.parse_args()

# %% load model
if not (os.path.isfile(f'{args.output_file}.npy')):
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # load segments
    with open(f"{args.input_path}time_stamps.json", "r", encoding="utf-8") as f:
        segments = json.load(f)

    def embed_frames(frames):
        inputs = clip_processor(images=frames, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = clip_model.get_image_features(**inputs)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return np.array(embeddings)

    frames = []
    i = 1
    while (os.path.isfile(args.input_path+f'scene_{i:02d}.jpg')):
        frame = Image.open(args.input_path+f'scene_{i:02d}.jpg').convert("RGB")
        frames.append(frame.copy())
        i += 1
    embeddings = embed_frames(frames)

    # save embeddings
    np.save(f'{args.output_file}.npy', embeddings)

    # postgreSQL
    def store_db(embeddings, segments):
        import psycopg2
        from pgvector.psycopg2 import register_vector

        # %% Connect to PostgreSQL and register vector
        conn = psycopg2.connect(
            dbname='embeddings',
            user='video_rag',
            password='rag',
            host='localhost',
            port='5432')
        
        cur = conn.cursor()
        cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

        # %% register the vector type with connection
        register_vector(conn)

        # %% create tables
        cur.execute('CREATE TABLE lecture_frames_embeddings (frame_id bigserial PRIMARY KEY, embedding vector(512), start_time FLOAT, end_time FLOAT)')

        # %% Insert records
        for embedding, segment in zip(embeddings, segments):
            cur.execute(
                "INSERT INTO lecture_frames_embeddings (embedding, start_time, end_time) VALUES (%s, %s, %s)",
                (list(embedding.astype(float)), float(segment['start']), float(segment['end']))
            )

        # %% Commit and close
        conn.commit()
        cur.close()
        conn.close()
        return
    store_db(embeddings, segments)
else:
    print('Frames already embedded. Step skipped.')