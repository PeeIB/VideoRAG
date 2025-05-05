# %% import libraries and modules
import json
from rank_bm25 import BM25Okapi

# load segments
with open(f"files/lecture_segments.json", "r", encoding="utf-8") as f:
    segments = json.load(f)
texts = [seg["text"] for seg in segments]

# %% lexical embeddings: BM25
tokenized_corpus = [text.split(" ") for text in texts]
bm25 = BM25Okapi(tokenized_corpus)

# %%
query = "windy London"
tokenized_query = query.split(" ")
bm25.get_top_n(tokenized_query, texts, n=1)
