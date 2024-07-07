import streamlit as st
import yt_dlp
import openai_whisper
import os
from io import BytesIO

# Function to download audio from YouTube video
def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': '%(id)s.%(ext)s',
        'noplaylist': True,
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_id = info_dict.get("id", None)
        filename = ydl.prepare_filename(info_dict).replace('.webm', '.mp3').replace('.m4a', '.mp3')
        ydl.download([url])
        return filename

# Function to transcribe audio using Whisper
def transcribe_audio(file_path):
    model = openai_whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

# Streamlit app
st.title("YouTube Video Transcription")

# Input YouTube URL
youtube_url = st.text_input("Enter YouTube video URL:")

if youtube_url:
    if st.button("Transcribe"):
        with st.spinner("Downloading audio..."):
            audio_file = download_audio(youtube_url)
        
        if audio_file:
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(audio_file)
            
            st.success("Transcription completed!")
            
            # Display transcription
            st.text_area("Transcription:", transcription, height=300)
            
            # Provide download link for the transcription
            b = BytesIO(transcription.encode())
            b.name = "transcription.txt"
            st.download_button(label="Download Transcription", data=b, file_name="transcription.txt", mime="text/plain")
            
            # Clean up downloaded audio file
            os.remove(audio_file)
