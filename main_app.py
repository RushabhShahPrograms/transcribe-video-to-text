import streamlit as st
import yt_dlp
import os
import whisper
import torch

# Function to download YouTube audio
def download_youtube_audio(url, download_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(download_path, '%(title)s.%(ext)s')
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp3"
    
    return audio_path, info['title'], info['thumbnail']

# Function to transcribe audio to text
def transcribe_audio(audio_path, model_size="base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(audio_path)
    return result["text"]

# Streamlit UI
st.title("YouTube Video to Text Transcriber")
video_url = st.text_input("Enter YouTube Video URL")
if video_url:
    with st.spinner("Downloading audio..."):
        download_path = "./temp"
        os.makedirs(download_path, exist_ok=True)
        audio_path, video_title, thumbnail_url = download_youtube_audio(video_url, download_path)
    
    st.image(thumbnail_url, caption=video_title)
    
    if st.button("Transcribe Audio"):
        with st.spinner("Transcribing audio to text..."):
            transcription = transcribe_audio(audio_path)
        
        st.text_area("Transcription", transcription, height=300)
        
        if st.download_button("Download Transcription as TXT", transcription, file_name="transcription.txt"):
            # Clean up temporary files
            os.remove(audio_path)
            st.success("Temporary files deleted!")
