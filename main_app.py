import streamlit as st
import whisper
from io import BytesIO

# Load Whisper model
model = whisper.load_model("base")

# Streamlit app
st.title("Audio to Text Transcription")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    with st.spinner("Transcribing audio..."):
        # Save the uploaded file temporarily
        with open("temp_audio", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Transcribe audio using Whisper
        result = model.transcribe("temp_audio")

        # Display transcription
        transcription = result["text"]
        st.text_area("Transcription:", transcription, height=300)

        # Provide download link for the transcription
        b = BytesIO(transcription.encode())
        b.name = "transcription.txt"
        st.download_button(label="Download Transcription", data=b, file_name="transcription.txt", mime="text/plain")
