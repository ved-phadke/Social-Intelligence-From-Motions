import streamlit as st
import plotly.express as px
import tempfile
import os
import torch
from video_frame import save_video_frame
from test import run_model
from PIL import Image

st.title('Emotion and Affect Tracking App')

uploaded_file = st.file_uploader("Choose a video file")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_file_path = tmp_file.name

    with tempfile.TemporaryDirectory() as frames_save_path:
        frames_path, frame_timestamps = save_video_frame(video_file_path, frames_save_path)
    
        device = torch.device('cpu')
        predictions = run_model(frames_path, device)

        valence_data = predictions[:, 0]
        arousal_data = predictions[:, 1]

        # Create a DataFrame for plotting
        import pandas as pd
        frame_timestamps = frame_timestamps[::5]
        data = pd.DataFrame({
            'Frame': range(len(valence_data)), 
            'Valence': valence_data, 
            'Arousal': arousal_data, 
            'Timestamp': frame_timestamps
        })

        # graphs
        fig_valence = px.line(data, x='Frame', y='Valence', title='Valence over Frames')
        fig_arousal = px.line(data, x='Frame', y='Arousal', title='Arousal over Frames')
        st.plotly_chart(fig_valence, use_container_width=True)
        st.plotly_chart(fig_arousal, use_container_width=True)

        # Frame selector
        selected_frame = st.selectbox('Select a frame to view:', data['Frame'])
        frame_path = os.path.join(frames_path, f"{selected_frame}.jpg")
        image = Image.open(frame_path)
        st.image(image, caption=f"Frame at {data['Timestamp'][selected_frame]:.2f} seconds")

    os.remove(video_file_path)
