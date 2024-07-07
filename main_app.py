import streamlit as st
import yt_dlp
import whisper
import os
import torch

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("large").to(device)

model = load_whisper_model()

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio.%(ext)s'
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def transcribe_audio():
    result = model.transcribe("audio.mp3")
    return result["text"]

def main():
    st.title("YouTube Video Transcription App")

    url = st.text_input("Enter YouTube video URL:")

    if st.button("Transcribe"):
        if url:
            with st.spinner("Downloading audio..."):
                download_audio(url)
            
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio()
            
            st.success("Transcription complete!")
            
            # Create a download button for the transcription
            st.download_button(
                label="Download Transcription",
                data=transcription,
                file_name="transcription.txt",
                mime="text/plain"
            )

            # Clean up
            if os.path.exists("audio.mp3"):
                os.remove("audio.mp3")

        else:
            st.warning("Please enter a valid YouTube URL.")

if __name__ == "__main__":
    main()
