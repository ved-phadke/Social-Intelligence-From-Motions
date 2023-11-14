import streamlit as st
import plotly.express as px
import tempfile
import os
import torch
from video_frame import save_video_frame
from test import run_model

st.title('Emotion and Affect Tracking App')

uploaded_file = st.file_uploader("Choose a video file")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_file_path = tmp_file.name

    with tempfile.TemporaryDirectory() as frames_save_path:
        frames_path = save_video_frame(video_file_path, frames_save_path)
    
        device = torch.device('cpu')

        predictions = run_model(frames_path, device)

        valence_data = predictions[:, 0]
        arousal_data = predictions[:, 1]

        fig_valence = px.line(valence_data, title='Valence over Frames')
        fig_arousal = px.line(arousal_data, title='Arousal over Frames')

        st.plotly_chart(fig_valence)
        st.plotly_chart(fig_arousal)

    os.remove(video_file_path)
