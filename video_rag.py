'''
video_link = 'https://www.youtube.com/watch?v=dARr3lGKwk8'
file_name = 'lecture'
'''
# import libraries and modules
import streamlit as st
import subprocess
from inference import FAISS, IVFFLAT, HNSW, TFIDF, BM25, CLIP

# Session state to handle phase transition
if 'stage' not in st.session_state:
    st.session_state.stage = 'input'

# ---------- Stage 1: Input Form ----------
if st.session_state.stage == 'input':
    st.title('Video Processing App')

    video_link = st.text_input('Enter YouTube video link')
    file_name = st.text_input('Enter file name (no extension)')
    st.session_state.file_name = file_name
    if st.button('Submit'):
        if video_link and file_name:
            # Run your processing script here
            subprocess.run(['python3', 'process_video.py',
                            '--video_link', video_link,
                            '--file_name', file_name])

            # Move to next stage
            st.session_state.stage = 'method_select'
            st.rerun()
        else:
            st.warning('Please fill in both fields.')

# ---------- Stage 2: Method Selection ----------
elif st.session_state.stage == 'method_select':
    st.title(f'Video RAG - {st.session_state.file_name}.mp4')

    method = st.selectbox('Choose a search strategy:', ['FAISS', 'IVFFLAT', 'HNSW', 'TF-IDF', 'BM25', 'CLIP'])


    # Input box for user query
    query = st.text_input('Enter your question about the video content')

    if st.button('Proceed'):
        if query:
            st.success(f'You selected: {method}\n\nQuery: {query}')
            if method == 'FAISS':
                similarity, start_time = FAISS(st.session_state.file_name, query)
                if(similarity>0.7):
                    st.video(f'files/{st.session_state.file_name}.mp4', start_time=start_time)
                else:
                    st.warning('The video does not include the answer to your question.')


            if method == 'IVFFLAT':
                similarity, start_time = IVFFLAT(st.session_state.file_name, query)
                if(similarity>0.7):
                    st.video(f'files/{st.session_state.file_name}.mp4', start_time=start_time)
                else:
                    st.warning('The video does not include the answer to your question.')


            if method == 'HNSW':
                similarity, start_time = HNSW(st.session_state.file_name, query)
                if(similarity>0.7):
                    st.video(f'files/{st.session_state.file_name}.mp4', start_time=start_time)
                else:
                    st.warning('The video does not include the answer to your question.')


            if method == 'TF-IDF':
                similarity, start_time = TFIDF(st.session_state.file_name, query)
                if(similarity>0.2):
                    st.video(f'files/{st.session_state.file_name}.mp4', start_time=start_time)
                else:
                    st.warning('The video does not include the answer to your question.')


            if method == 'BM25':
                score, start_time = BM25(st.session_state.file_name, query)
                if(score>5):
                    st.video(f'files/{st.session_state.file_name}.mp4', start_time=start_time)
                else:
                    st.warning('The video does not include the answer to your question.')


            if method == 'CLIP':
                similarity, start_time = CLIP(st.session_state.file_name, query)
                if(similarity>0.24):
                    st.video(f'files/{st.session_state.file_name}.mp4', start_time=start_time)
                else:
                    st.warning('The video does not include the answer to your question.')

        else:
            st.warning("Please enter a query before proceeding.")
