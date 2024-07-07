import streamlit as st
import whisper
from io import BytesIO
import os
import tempfile

# Load Whisper model
@st.cache(allow_output_mutation=True)
def load_model():
    return whisper.load_model("tiny")

model = load_model()

# Streamlit app
st.title("Audio to Text Transcription")

# Initialize session state
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'transcribed' not in st.session_state:
    st.session_state.transcribed = False

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None and not st.session_state.transcribed:
    with st.spinner("Transcribing audio..."):
        try:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_audio:
                temp_audio.write(uploaded_file.getbuffer())
                temp_audio_path = temp_audio.name

            # Transcribe audio using Whisper
            result = model.transcribe(temp_audio_path)

            # Store transcription in session state
            st.session_state.transcription = result["text"]
            st.session_state.transcribed = True

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        finally:
            # Clean up temporary audio file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

# Display transcription and download button if available
if st.session_state.transcription:
    st.text_area("Transcription:", st.session_state.transcription, height=300)

    # Provide download link for the transcription
    b = BytesIO(st.session_state.transcription.encode())
    b.name = "transcription.txt"
    st.download_button(label="Download Transcription", data=b, file_name="transcription.txt", mime="text/plain")
