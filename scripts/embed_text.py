# %% import libraries and modules
import os
import json
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# %% parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file_name')
parser.add_argument('--input_file')
parser.add_argument('--output_file')
args = parser.parse_args()

# %% load embedding model
if not (os.path.isfile(f'{args.output_file}.npy')):
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    model.eval()

    # load segments
    with open(f"{args.input_file}.json", "r", encoding="utf-8") as f:
        segments = json.load(f)
    texts = [seg["text"] for seg in segments]

    # Embed texts
    def embed_text(text_list):
        inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    embeddings = np.array(embed_text(texts))

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
        cur.execute('CREATE TABLE lecture_text_embeddings (chunk_id bigserial PRIMARY KEY, text TEXT, embedding vector(384), start_time FLOAT, end_time FLOAT)')
        # %% Insert records
        for embedding, segment in zip(embeddings, segments):
            cur.execute(
                "INSERT INTO lecture_text_embeddings (text, embedding, start_time, end_time) VALUES (%s, %s, %s, %s)",
                (segment["text"], list(embedding.astype(float)), float(segment["start"]), float(segment["end"])))
        # %% Commit and close
        conn.commit()
        cur.close()
        conn.close()
        return
    store_db(embeddings, segments)
else:
    print('Chunks already embedded. Step skipped.')

