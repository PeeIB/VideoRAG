# %% import libraries and modules
import json
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

# %% Load embeddings and metadata
text_embeddings = np.load("../files/lecture_text_embeddings.npy")
frames_embeddings = np.load("../files/lecture_frames_embeddings.npy")

with open("../files/lecture_segments.json", "r", encoding="utf-8") as f:
    segments = json.load(f) 

with open("../files/lecture_frames/time_stamps.json", "r", encoding="utf-8") as f:
    frames_time_stamps = json.load(f)  


# %% Connect to PostgreSQL and register vector
conn = psycopg2.connect(
    dbname='embeddings',
    user='video_rag',
    password='rag',
    host='localhost',
    port='5432'
)
cur = conn.cursor()
cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

# %% register the vector type with connection
register_vector(conn)

# %% create tables
cur.execute('CREATE TABLE lecture_text_embeddings (chunk_id bigserial PRIMARY KEY, text TEXT, embedding vector(384), start_time FLOAT, end_time FLOAT)')
cur.execute('CREATE TABLE lecture_frames_embeddings (frame_id bigserial PRIMARY KEY, embedding vector(512), start_time FLOAT, end_time FLOAT)')

# %% Insert records
for embedding, segment in zip(text_embeddings, segments):
    cur.execute(
        "INSERT INTO lecture_text_embeddings (text, embedding, start_time, end_time) VALUES (%s, %s, %s, %s)",
        (segment["text"], list(embedding.astype(float)), float(segment["start"]), float(segment["end"]))
    )
    
for embedding, time_stamps in zip(frames_embeddings, frames_time_stamps):
    cur.execute(
        "INSERT INTO lecture_frames_embeddings (embedding, start_time, end_time) VALUES (%s, %s, %s)",
        (list(embedding.astype(float)), float(time_stamps['start']), float(time_stamps['end']))
    )

# %% Commit and close
conn.commit()
cur.close()
conn.close()


# %%
