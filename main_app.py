import streamlit as st
from pytube import YouTube
import subprocess
import os
import whisper
from moviepy.editor import VideoFileClip
import torch

# Function to download YouTube video
def download_youtube_video(url, download_path):
    yt = YouTube(url)
    stream = yt.streams.get_lowest_resolution()
    video_path = stream.download(download_path)
    return video_path, yt.title, yt.thumbnail_url

# Function to convert video to audio
def convert_to_audio(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path)
    video_clip.close()

# Function to transcribe audio to text
def transcribe_audio(audio_path, model_size="base"):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(audio_path)
    return result["text"]

# Streamlit UI
st.title("YouTube Video to Text Transcriber")
video_url = st.text_input("Enter YouTube Video URL")
if video_url:
    with st.spinner("Downloading video..."):
        download_path = "./temp"
        os.makedirs(download_path, exist_ok=True)
        video_path, video_title, thumbnail_url = download_youtube_video(video_url, download_path)
    
    st.image(thumbnail_url, caption=video_title)
    
    if st.button("Transcribe Video"):
        audio_path = os.path.join(download_path, "audio.mp3")
        with st.spinner("Converting video to audio..."):
            convert_to_audio(video_path, audio_path)
        
        with st.spinner("Transcribing audio to text..."):
            transcription = transcribe_audio(audio_path)
        
        st.text_area("Transcription", transcription, height=300)
        
        if st.download_button("Download Transcription as TXT", transcription, file_name="transcription.txt"):
            # Clean up temporary files
            os.remove(video_path)
            os.remove(audio_path)
            st.success("Temporary files deleted!")

# Run the Streamlit app by executing `streamlit run script_name.py` from the command line
