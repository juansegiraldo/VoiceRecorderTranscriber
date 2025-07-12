import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import shutil
from pydub import AudioSegment
import math
import time
import io

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Voice Transcriber",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def split_audio_file(file_path: str, max_size_mb: int = 24) -> list[str]:
    """Split a large audio file into smaller chunks that fit within API limits."""
    # Convert MB to bytes (leaving some buffer)
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # If file is small enough, return original path
    if file_size <= max_size_bytes:
        return [file_path]
    
    # Calculate how many chunks we need
    num_chunks = math.ceil(file_size / max_size_bytes)
    chunk_duration = len(audio) // num_chunks
    
    # Create temporary directory for chunks
    temp_dir = tempfile.mkdtemp()
    chunk_paths = []
    
    try:
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = start_time + chunk_duration if i < num_chunks - 1 else len(audio)
            
            # Extract chunk
            chunk = audio[start_time:end_time]
            
            # Save chunk to temporary file
            chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.mp3")
            chunk.export(chunk_path, format="mp3")
            chunk_paths.append(chunk_path)
            
        return chunk_paths
    except Exception as e:
        # Clean up temp directory on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

def transcribe_file(path: str) -> str:
    """Upload an audio file to OpenAI and return the transcription text."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)

    with open(path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

def transcribe_large_file(file_path: str, progress_bar=None, status_text=None) -> str:
    """Transcribe a large audio file by splitting it into chunks and combining results."""
    if status_text:
        status_text.text("🔍 Analizando archivo...")
    
    # Split the file into chunks
    chunk_paths = split_audio_file(file_path)
    
    if len(chunk_paths) == 1:
        # File is small enough, process normally
        if status_text:
            status_text.text("🎤 Transcribiendo archivo...")
        return transcribe_file(file_path)
    
    if status_text:
        status_text.text(f"📦 Archivo dividido en {len(chunk_paths)} chunks")
    
    # Process each chunk
    transcriptions = []
    temp_dir = os.path.dirname(chunk_paths[0]) if len(chunk_paths) > 1 else None
    
    try:
        for i, chunk_path in enumerate(chunk_paths, 1):
            if status_text:
                status_text.text(f"🎤 Procesando chunk {i}/{len(chunk_paths)}...")
            if progress_bar:
                progress_bar.progress(i / len(chunk_paths))
            
            try:
                chunk_transcription = transcribe_file(chunk_path)
                transcriptions.append(chunk_transcription)
                if status_text:
                    status_text.text(f"✅ Chunk {i} procesado")
            except Exception as e:
                if status_text:
                    status_text.text(f"❌ Error procesando chunk {i}: {str(e)}")
                # Continue with other chunks even if one fails
                transcriptions.append(f"[Error en chunk {i}: {str(e)}]")
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Combine all transcriptions
    combined_transcription = " ".join(transcriptions)
    return combined_transcription

def main():
    # Header
    st.markdown('<h1 class="main-header">🎤 Voice Transcriber</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transcribe audio files using OpenAI\'s Whisper API</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key. You can also set it in a .env file."
        )
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # File size limit
        max_file_size = st.slider(
            "Max file size (MB) for chunking",
            min_value=10,
            max_value=50,
            value=24,
            help="Files larger than this will be split into chunks"
        )
        
        st.divider()
        
        # Supported formats info
        st.subheader("📁 Supported Formats")
        st.markdown("""
        - MP3
        - WAV
        - M4A
        - FLAC
        - OGG
        - AAC
        - WMA
        """)
        
        st.divider()
        
        # Instructions
        st.subheader("📋 Instructions")
        st.markdown("""
        1. Upload your audio file
        2. Wait for processing
        3. Download the transcription
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎵 Upload Audio File")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac', 'wma'],
            help="Select an audio file to transcribe"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_size = len(uploaded_file.getvalue())
            file_size_mb = file_size / (1024 * 1024)
            
            st.info(f"📄 File: {uploaded_file.name}")
            st.info(f"📏 Size: {file_size_mb:.2f} MB")
            
            # Transcribe button
            if st.button("🎤 Start Transcription", type="primary"):
                # Check API key
                if not os.environ.get("OPENAI_API_KEY"):
                    st.error("❌ OpenAI API key not found. Please enter it in the sidebar.")
                    return
                
                # Create progress elements
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Transcribe the file
                    transcription = transcribe_large_file(
                        tmp_file_path, 
                        progress_bar=progress_bar, 
                        status_text=status_text
                    )
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    # Update progress to 100%
                    progress_bar.progress(1.0)
                    status_text.text("✅ Transcription completed!")
                    
                    # Store transcription in session state
                    st.session_state.transcription = transcription
                    st.session_state.filename = uploaded_file.name
                    
                    st.success("🎉 Transcription completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during transcription: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
    
    with col2:
        st.header("📝 Results")
        
        if 'transcription' in st.session_state:
            # Display transcription
            st.subheader(f"Transcription: {st.session_state.filename}")
            
            # Text area for transcription
            transcription_text = st.text_area(
                "Transcription",
                value=st.session_state.transcription,
                height=300,
                help="The transcribed text from your audio file"
            )
            
            # Download button
            if transcription_text:
                # Create download data
                download_data = transcription_text.encode('utf-8')
                
                st.download_button(
                    label="📥 Download Transcription",
                    data=download_data,
                    file_name=f"{st.session_state.filename.split('.')[0]}_transcript.txt",
                    mime="text/plain",
                    help="Download the transcription as a text file"
                )
                
                # Copy to clipboard button
                if st.button("📋 Copy to Clipboard"):
                    st.write("📋 Copied to clipboard!")
                    st.code(transcription_text)
        else:
            st.info("👆 Upload a file and start transcription to see results here")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Powered by OpenAI Whisper API | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 