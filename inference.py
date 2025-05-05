# %% FAISS
def FAISS(file_name, query):
    import os
    os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
    import json
    import faiss
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F

    # load model
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    model.eval()

    # Embed texts
    input = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**input)
    query_embedding = output.last_hidden_state[:, 0, :]
    query_embedding = F.normalize(query_embedding, p=2, dim=1)

    # load embeddings
    embeddings = np.load(f'files/{file_name}_text_embeddings.npy')
    dim = embeddings.shape[1]
    faiss_index_text = faiss.IndexFlatIP(dim)  # use inner product for cosine
    faiss_index_text.add(embeddings)

    # find approximate nearest neighbors
    D, I = faiss_index_text.search(query_embedding, k=1)
    I = np.squeeze(I)
    # load text segments
    with open(f"files/{file_name}_segments.json", "r", encoding="utf-8") as f:
        segments = json.load(f)
    start_times = [seg["start"] for seg in segments]

    return D, start_times[I]

# %% IVFFLAT
def IVFFLAT(file_name, query):
    import psycopg2
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn.functional as F
    
    # load model
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    model.eval()

    # Embed texts
    input = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**input)
    query_embedding = output.last_hidden_state[:, 0, :]
    query_embedding = np.array(F.normalize(query_embedding, p=2, dim=1)).squeeze().astype('float')

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname='embeddings',
        user='video_rag',
        password='rag',
        host='localhost',
        port='5432'
    )
    cur = conn.cursor()

    # Create IVFFLAT index if it doesn't exist
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS {file_name}_ivfflat_idx
        ON {file_name}_text_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10);
    """)
    cur.execute('SET ivfflat.probes = 5;')

    cur.execute(
        "INSERT INTO lecture_text_embeddings (chunk_id, embedding) VALUES (%s, %s)",
        (-1, list(query_embedding))
    )

    # Run similarity search
    cur.execute("""
        SELECT start_time, embedding <=> (SELECT embedding FROM lecture_text_embeddings WHERE chunk_id = -1)
                FROM lecture_text_embeddings 
                WHERE chunk_id != -1 
                ORDER BY embedding <-> (SELECT embedding FROM lecture_text_embeddings WHERE chunk_id = -1) 
                LIMIT 1;
    """)
    results = cur.fetchall()

    cur.close()
    conn.close()
    return 1-results[0][1], results[0][0]

# %% HNSW
def HNSW(file_name, query):
    import psycopg2
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn.functional as F
    
    # load model
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    model.eval()

    # Embed texts
    input = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**input)
    query_embedding = output.last_hidden_state[:, 0, :]
    query_embedding = np.array(F.normalize(query_embedding, p=2, dim=1)).squeeze().astype('float')

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname='embeddings',
        user='video_rag',
        password='rag',
        host='localhost',
        port='5432'
    )
    cur = conn.cursor()

    # Create IVFFLAT index if it doesn't exist
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS {file_name}_hnsw_idx
        ON {file_name}_text_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 4, ef_construction = 16);
    """)

    cur.execute(
        "INSERT INTO lecture_text_embeddings (chunk_id, embedding) VALUES (%s, %s)",
        (-1, list(query_embedding))
    )

    # Run similarity search
    cur.execute("""
        SELECT start_time, embedding <=> (SELECT embedding FROM lecture_text_embeddings WHERE chunk_id = -1)
                FROM lecture_text_embeddings 
                WHERE chunk_id != -1 
                ORDER BY embedding <-> (SELECT embedding FROM lecture_text_embeddings WHERE chunk_id = -1) 
                LIMIT 1;
    """)
    results = cur.fetchall()

    cur.close()
    conn.close()
    return 1-results[0][1], results[0][0]

# %% TF-IDF
def TFIDF(file_name, query):
    import json
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    # load text chunks
    with open(f"files/{file_name}_segments.json", "r", encoding="utf-8") as f:
        segments = json.load(f)
    texts = [seg["text"] for seg in segments]
    texts.append(query)
    start_times = [seg["start"] for seg in segments]

    # run tf-idf
    tf_idf = TfidfVectorizer()
    tf_idf_matrix = tf_idf.fit_transform(texts).todense()
    embeddings = tf_idf_matrix[:-1]
    query_embedding = tf_idf_matrix[-1].reshape(-1, 1)
    
    # normalize
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    I = np.argmax(embeddings @ query_embedding)
    similarity = np.max(embeddings @ query_embedding)
    return similarity, start_times[I]

# %% BM25
def BM25(file_name, query):
    import json
    from rank_bm25 import BM25Okapi
    import numpy as np
    
    # load segments
    with open(f"files/{file_name}_segments.json", "r", encoding="utf-8") as f:
        segments = json.load(f)
    texts = [seg["text"] for seg in segments]
    start_times = [seg["start"] for seg in segments]
    
    # lexical embeddings: BM25
    tokenized_corpus = [text.split(" ") for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.split(" ")

    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)
    I = np.argmax(scores)
    score = np.max(scores)
    return score, start_times[I]

# %% CLIP
def CLIP(file_name, query):
    import json
    from transformers import CLIPModel, AutoProcessor
    import torch
    import torch.nn.functional as F
    import numpy as np

    # load model and processor (same as for image embedding)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # tokenize and embed query
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = clip_model.get_text_features(**inputs)
    query_embedding = F.normalize(query_embedding, p=2, dim=1).cpu().numpy().squeeze()

    # load embeddings
    embeddings = np.load(f'files/{file_name}_frames_embeddings.npy')
    
    # normalize
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    I = np.argmax(embeddings @ query_embedding)
    similarity = np.max(embeddings @ query_embedding)

    # load start times
    with open(f"files/{file_name}_frames/time_stamps.json", "r", encoding="utf-8") as f:
        segments = json.load(f)
    start_times = [seg["start"] for seg in segments]

    return similarity, start_times[I]

file_name='lecture'
query='token placement'