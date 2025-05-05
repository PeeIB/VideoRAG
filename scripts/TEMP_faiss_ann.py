# %% import libraries and modules
import faiss
import numpy as np

# %% load embeddings
embeddings = np.load('files/lecture_text_embeddings.npy')
dim = embeddings.shape[1]
faiss_index_text = faiss.IndexFlatIP(dim)  # use inner product for cosine
faiss_index_text.add(embeddings)

# %% find approximate nearest neighbors
query_emb = np.ones(dim, dtype='float32').reshape(1, -1)
D, I = faiss_index_text.search(query_emb, k=5)