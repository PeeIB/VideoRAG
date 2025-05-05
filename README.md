# 🎥📚 Lecture Video Embedding & Retrieval System

This project processes YouTube lecture videos and enables semantic and lexical search across video frames and transcript segments using embedding models like **CLIP** and **BGE**.

---

## 🔧 Features

- ✅ Download and process YouTube videos.
- 🖼️ Extract video frames and embed them using OpenAI’s CLIP.
- 📝 Transcribe videos using OpenAI Whisper and embed text chunks using BAAI’s BGE.
- 📦 Store embeddings in PostgreSQL with `pgvector`.
- 🔍 Enable multimodal search: find frames and text semantically similar to a query.
- 🔄 Integrate with BM25, TF-IDF, FAISS, and other retrieval methods (via Streamlit app, WIP).

---

## 📁 Project Structure

```bash
.
├── download_video.py          # Downloads video and generates transcript
├── extract_frames.py          # Extracts frames from video (not shown here)
├── embed_frames.py            # Embeds video frames using CLIP
├── embed_text.py              # Embeds transcript segments using BGE
├── data/
│   ├── video.mp4
│   ├── transcript.json
│   ├── time_stamps.json
│   └── (frame images, if needed)
├── outputs/
│   ├── clip_frame_embeddings.npy
│   └── bge_text_embeddings.npy
├── app/                       # Streamlit app (optional)
└── README.md

