import streamlit as st
import os
from openai import OpenAI
import requests
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import shutil
import math
import time
import io
import logging
import imageio_ffmpeg

# Patch pydub to use imageio-ffmpeg's bundled ffmpeg/ffprobe BEFORE importing AudioSegment
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
print(f"Using FFmpeg from: {ffmpeg_path}")
from pydub.utils import which
from pydub import AudioSegment
AudioSegment.converter = ffmpeg_path
AudioSegment.ffmpeg = ffmpeg_path
print(f"Patched AudioSegment.converter to: {AudioSegment.converter}")
print(f"Patched AudioSegment.ffmpeg to: {AudioSegment.ffmpeg}")

# Set up logging for conversion
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Check for FFmpeg availability
def check_ffmpeg():
    """Check if FFmpeg is available for audio processing."""
    try:
        from pydub import AudioSegment
        # Try to load a simple audio file to test FFmpeg
        test_audio = AudioSegment.silent(duration=100)
        return True
    except Exception as e:
        if "ffprobe" in str(e) or "ffmpeg" in str(e):
            return False
        return True

# Check FFmpeg at startup
ffmpeg_available = check_ffmpeg()

def format_time(seconds):
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def format_time_ms(milliseconds):
    """Convert milliseconds to MM:SS format."""
    return format_time(milliseconds / 1000)

def trim_audio_file(input_path: str, start_time_ms: int, end_time_ms: int, output_path: str | None = None) -> str:
    """
    Trim an audio file to the specified time range.
    
    Args:
        input_path (str): Path to the input audio file
        start_time_ms (int): Start time in milliseconds
        end_time_ms (int): End time in milliseconds
        output_path (str, optional): Path for the output file. If None, 
                                   will create a temporary file
    
    Returns:
        str: Path to the trimmed audio file
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(input_path)
        
        # Ensure times are within bounds
        start_time_ms = max(0, min(start_time_ms, len(audio)))
        end_time_ms = max(start_time_ms, min(end_time_ms, len(audio)))
        
        # Trim the audio
        trimmed_audio = audio[start_time_ms:end_time_ms]
        
        # Determine output path
        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                output_path = tmp_file.name
        
        # Export the trimmed audio
        trimmed_audio.export(output_path, format="mp3")
        
        logger.info(f"Successfully trimmed {input_path} from {format_time_ms(start_time_ms)} to {format_time_ms(end_time_ms)}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error trimming {input_path}: {str(e)}")
        raise e

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
    max_size_bytes = max_size_mb * 1024 * 1024
    audio = AudioSegment.from_file(file_path)
    file_size = os.path.getsize(file_path)
    if file_size <= max_size_bytes:
        return [file_path]
    num_chunks = math.ceil(file_size / max_size_bytes)
    chunk_duration = len(audio) // num_chunks
    temp_dir = tempfile.mkdtemp()
    chunk_paths = []
    try:
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = start_time + chunk_duration if i < num_chunks - 1 else len(audio)
            chunk = audio[start_time:end_time]
            chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.mp3")
            chunk.export(chunk_path, format="mp3")
            chunk_paths.append(chunk_path)
        return chunk_paths
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

def transcribe_with_openai(path: str) -> str:
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

def transcribe_with_deepgram(path: str) -> str:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPGRAM_API_KEY environment variable not set")
    
    # Try with language detection and better parameters
    url_default = "https://api.deepgram.com/v1/listen?smart_format=true&punctuate=true&diarize=false&language=es&model=base"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/mp3",
        "Accept": "application/json",
    }
    with open(path, "rb") as f:
        audio_data = f.read()
    response = requests.post(url_default, headers=headers, data=audio_data)
    if not response.ok:
        print("Deepgram error response (Spanish model):", response.text)
        return f"Deepgram error (Spanish model): {response.text}"
    dg = response.json()
    print("Deepgram raw response (Spanish model):", dg)
    transcript = dg.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
    if transcript and len(transcript.strip()) > 10:  # Only return if we have substantial text
        return transcript
    
    # If Spanish model didn't work well, try with language detection
    url_auto = "https://api.deepgram.com/v1/listen?smart_format=true&punctuate=true&diarize=false&detect_language=true"
    response2 = requests.post(url_auto, headers=headers, data=audio_data)
    if not response2.ok:
        print("Deepgram error response (auto-detect):", response2.text)
        return f"Deepgram error (auto-detect): {response2.text}"
    dg2 = response2.json()
    print("Deepgram raw response (auto-detect):", dg2)
    transcript2 = dg2.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
    if transcript2 and len(transcript2.strip()) > 10:
        return transcript2
    
    # If still no good result, try the original default model
    url_original = "https://api.deepgram.com/v1/listen?smart_format=true"
    response3 = requests.post(url_original, headers=headers, data=audio_data)
    if not response3.ok:
        print("Deepgram error response (original):", response3.text)
        return f"Deepgram error (original): {response3.text}"
    dg3 = response3.json()
    print("Deepgram raw response (original):", dg3)
    transcript3 = dg3.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
    if transcript3:
        return transcript3
    
    # If all attempts failed, return debug info
    return f"Deepgram failed to transcribe properly. Responses:\nSpanish: {dg}\nAuto-detect: {dg2}\nOriginal: {dg3}"

def transcribe_file(path: str, model: str) -> str:
    if model == "OpenAI Whisper":
        return transcribe_with_openai(path)
    elif model == "Deepgram":
        return transcribe_with_deepgram(path)
    else:
        raise ValueError(f"Unknown model: {model}")

def transcribe_large_file(file_path: str, model: str, progress_bar=None, status_text=None) -> str:
    if status_text:
        status_text.text("🔍 Analyzing file...")
    chunk_paths = split_audio_file(file_path)
    if len(chunk_paths) == 1:
        if status_text:
            status_text.text("🎤 Transcribing file...")
        return transcribe_file(file_path, model)
    if status_text:
        status_text.text(f"📦 File split into {len(chunk_paths)} chunks")
    transcriptions = []
    temp_dir = os.path.dirname(chunk_paths[0]) if len(chunk_paths) > 1 else None
    try:
        for i, chunk_path in enumerate(chunk_paths, 1):
            if status_text:
                status_text.text(f"🎤 Processing chunk {i}/{len(chunk_paths)}...")
            if progress_bar:
                progress_bar.progress(i / len(chunk_paths))
            try:
                chunk_transcription = transcribe_file(chunk_path, model)
                transcriptions.append(chunk_transcription)
                if status_text:
                    status_text.text(f"✅ Chunk {i} processed")
            except Exception as e:
                if status_text:
                    status_text.text(f"❌ Error processing chunk {i}: {str(e)}")
                transcriptions.append(f"[Error in chunk {i}: {str(e)}]")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    combined_transcription = " ".join(transcriptions)
    return combined_transcription

def convert_m4a_to_mp3(input_path: str, output_path: str | None = None, bitrate: str = "192k") -> str:
    """
    Convert a single M4A file to MP3 format.
    
    Args:
        input_path (str): Path to the input M4A file
        output_path (str, optional): Path for the output MP3 file. If None, 
                                   will use the same name with .mp3 extension
        bitrate (str): MP3 bitrate (default: "192k")
    
    Returns:
        str: Path to the converted MP3 file
    """
    try:
        input_file = Path(input_path)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        if not input_file.suffix.lower() == '.m4a':
            logger.warning(f"File {input_path} doesn't have .m4a extension")
        
        # Determine output path
        if output_path is None:
            output_file = input_file.with_suffix('.mp3')
        else:
            output_file = Path(output_path)
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting {input_path} to {output_file}")
        
        # Check if FFmpeg is available before attempting conversion
        if not ffmpeg_available:
            raise Exception("M4A conversion is not available on this server. Please convert your M4A files to MP3 or WAV format before uploading. You can use online converters or the standalone converter script in the scripts/ folder.")
        
        # Load the audio file
        audio = AudioSegment.from_file(str(input_file), format="m4a")
        
        # Export as MP3
        audio.export(str(output_file), format="mp3", bitrate=bitrate)
        
        logger.info(f"Successfully converted {input_path} to {output_file}")
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error converting {input_path}: {str(e)}")
        if "ffprobe" in str(e) or "ffmpeg" in str(e):
            raise Exception("M4A conversion is not available on this server. Please convert your M4A files to MP3 or WAV format before uploading.")
        raise e

def main():
    st.markdown(
        '''<style>
        body, .main, .block-container {
            background: #fafbfc !important;
        }
        .block-container {
            max-width: 720px !important; /* Increased from 540px to ~33% wider */
            margin-left: auto;
            margin-right: auto;
        }
        @media (max-width: 900px) {
            .block-container {
                max-width: 98vw !important;
                padding: 0.5rem !important;
            }
        }
        .mobile-card {
            background: #fff;
            border-radius: 1.5rem;
            box-shadow: 0 4px 24px rgba(0,0,0,0.07);
            padding: 2rem 1.2rem 1.5rem 1.2rem;
            margin-bottom: 1.5rem;
        }
        .mobile-header {
            font-size: 2.2rem;
            font-weight: 800;
            text-align: left;
            margin-bottom: 0.2rem;
            letter-spacing: -1px;
        }
        .mobile-sub {
            color: #888;
            font-size: 1.1rem;
            margin-bottom: 1.2rem;
        }
        .mobile-search {
            display: flex;
            align-items: center;
            background: #f3f4f6;
            border-radius: 1.2rem;
            padding: 0.7rem 1.2rem;
            margin-bottom: 1.5rem;
            border: 1px solid #e5e7eb;
        }
        .mobile-search input {
            border: none;
            background: transparent;
            outline: none;
            font-size: 1.1rem;
            width: 100%;
        }
        .mobile-section-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .mobile-instructions li {
            margin-bottom: 0.5rem;
        }
        @media (max-width: 600px) {
            .block-container {
                max-width: 100% !important;
                padding: 0.5rem !important;
            }
            .mobile-card {
                padding: 1.2rem 0.5rem 1rem 0.5rem;
            }
            .mobile-header {
                font-size: 1.4rem;
            }
        }
        </style>''', unsafe_allow_html=True)

    # Logo and App Name (centered, no bubble)
    st.markdown('<div style="text-align:center;margin-bottom:1.2rem;"><span style="font-size:2.2rem;">🎤</span><div style="font-size:2rem;font-weight:700;margin-top:0.5rem;">Voice Transcriber</div></div>', unsafe_allow_html=True)

    # Transcription Section
    st.markdown('<div style="font-size:1.3rem;font-weight:600;margin-bottom:0.5rem;text-align:center;">Insert audio file</div>', unsafe_allow_html=True)
    # st.markdown('<div style="text-align:center;">Choose an audio file</div>', unsafe_allow_html=True)
    allowed_types = ['mp3', 'wav']
    if ffmpeg_available:
        allowed_types.append('m4a')
    uploaded_file = st.file_uploader(
        "",
        type=allowed_types,
        help="Select an audio file to transcribe" + (" (M4A conversion requires FFmpeg)" if not ffmpeg_available else ""),
        label_visibility="visible"
    )
    
    # Audio trimming section
    trim_settings = None
    if uploaded_file is not None:
        # Show file info
        #st.markdown(f'<div style="margin:0.5rem 0 0.7rem 0;font-size:1rem;color:#444;text-align:center;">📄 {uploaded_file.name} <span style="color:#888;font-size:0.95rem;">{len(uploaded_file.getvalue())/1024/1024:.1f}MB</span></div>', unsafe_allow_html=True)
        
        # Check if we need to load audio info (only if file changed or not cached)
        file_key = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.type}"
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_changed = (
            st.session_state.get('current_file_key') != file_key or
            (file_extension == 'm4a' and st.session_state.get('converted_mp3_file_key') != file_key)
        )
        if file_changed:
            try:
                # Create a temporary file for duration analysis
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                if file_extension == 'm4a':
                    with st.spinner("🔄 Converting M4A to MP3..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_tmp_file:
                            mp3_file_path = mp3_tmp_file.name
                        converted_path = convert_m4a_to_mp3(tmp_file_path, mp3_tmp_file.name)
                        os.unlink(tmp_file_path)
                        audio_path = converted_path
                        st.session_state.converted_mp3_path = audio_path
                        st.session_state.converted_mp3_file_key = file_key
                else:
                    audio_path = tmp_file_path
                    st.session_state.converted_mp3_path = None
                    st.session_state.converted_mp3_file_key = None
                audio = AudioSegment.from_file(audio_path)
                duration_ms = len(audio)
                duration_seconds = duration_ms / 1000
                if file_extension != 'm4a':
                    os.unlink(audio_path)
                st.session_state.audio_info = {
                    'duration_ms': duration_ms,
                    'duration_seconds': duration_seconds
                }
                st.session_state.current_file_key = file_key
            except Exception as e:
                st.error(f"❌ Error loading audio file: {str(e)}")
                trim_settings = None
                return
        audio_info = st.session_state.audio_info
        duration_seconds = audio_info['duration_seconds']
        
        # Show audio info
        st.markdown(f'<div style="font-size:0.9rem;color:#666;text-align:center;margin-bottom:1rem;">🎵 Duration: {format_time(duration_seconds)}</div>', unsafe_allow_html=True)
        
        # Trimming controls
        st.markdown('<div style="font-size:1.1rem;font-weight:600;margin-bottom:0.5rem;text-align:center;">Audio Trimming (Optional)</div>', unsafe_allow_html=True)
        
        # Use a single range slider for start and end time
        trim_range = st.slider(
            "Select audio range to transcribe",
            min_value=0.0,
            max_value=float(duration_seconds),
            value=(0.0, float(duration_seconds)),
            step=0.1,
            format="%.1f s",
            help="Select the portion of the audio to transcribe (start and end times)",
            key="trim_range_slider"
        )
        start_time, end_time = trim_range
        
        # Show trim preview
        trim_duration = end_time - start_time
        st.markdown(f'<div style="font-size:0.9rem;color:#666;text-align:center;margin-bottom:1rem;">✂️ Will transcribe: {format_time(start_time)} - {format_time(end_time)} ({format_time(trim_duration)} total)</div>', unsafe_allow_html=True)
        
        # Store trim settings
        trim_settings = {
            'start_time_ms': int(start_time * 1000),
            'end_time_ms': int(end_time * 1000),
            'duration_ms': audio_info['duration_ms']
        }
    
    st.markdown('<div style="margin-top:1.2rem;font-size:1.1rem;font-weight:600;text-align:center;">Results</div>', unsafe_allow_html=True)
    if uploaded_file is not None:
        if st.button("🎤 Start Transcription", type="primary"):
            model = st.session_state.get('model', 'Deepgram')
            if model == "OpenAI Whisper" and not os.environ.get("OPENAI_API_KEY"):
                st.error("❌ OpenAI API key not found. Please enter it in a .env file.")
                return
            elif model == "Deepgram" and not os.environ.get("DEEPGRAM_API_KEY"):
                st.error("❌ Deepgram API key not found. Please enter it in a .env file.")
                return
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                # Create a fresh temporary file for transcription
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                file_extension = uploaded_file.name.split('.')[-1].lower()
                temp_files_to_cleanup = [tmp_file_path]
                # Use cached MP3 if available
                audio_path: str = tmp_file_path  # Default
                if file_extension == 'm4a' and st.session_state.get('converted_mp3_path') and st.session_state.get('converted_mp3_file_key') == file_key:
                    if st.session_state.converted_mp3_path is not None:
                        audio_path = st.session_state.converted_mp3_path
                elif file_extension == 'm4a':
                    if status_text:
                        status_text.text("🔄 Converting M4A to MP3...")
                    time.sleep(0.5)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_tmp_file:
                        mp3_file_path = mp3_tmp_file.name
                    converted_path = convert_m4a_to_mp3(tmp_file_path, mp3_file_path)
                    temp_files_to_cleanup.append(converted_path)
                    audio_path = converted_path
                # Apply trimming if settings are provided
                if trim_settings and (trim_settings['start_time_ms'] > 0 or trim_settings['end_time_ms'] < trim_settings['duration_ms']):
                    if status_text:
                        status_text.text("✂️ Trimming audio...")
                    if audio_path is not None:
                        trimmed_path = trim_audio_file(
                            audio_path,
                            trim_settings['start_time_ms'],
                            trim_settings['end_time_ms']
                        )
                        temp_files_to_cleanup.append(trimmed_path)
                        audio_path = trimmed_path
                if audio_path is not None:
                    transcription = transcribe_large_file(
                        audio_path, 
                        st.session_state.get('model', 'Deepgram'),
                        progress_bar=progress_bar, 
                        status_text=status_text,
                    )
                else:
                    st.error("❌ Internal error: audio_path is None.")
                    return
                # Clean up all temp files (including cached mp3 after use)
                for temp_file in temp_files_to_cleanup:
                    try:
                        if temp_file and isinstance(temp_file, str) and os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except Exception as e:
                        logger.warning(f"Could not delete temp file {temp_file}: {e}")
                # If we used the cached mp3, clean it up now
                if file_extension == 'm4a' and st.session_state.get('converted_mp3_path') and st.session_state.get('converted_mp3_file_key') == file_key:
                    try:
                        cached_mp3 = st.session_state.converted_mp3_path
                        if cached_mp3 and isinstance(cached_mp3, str) and os.path.exists(cached_mp3):
                            os.unlink(cached_mp3)
                    except Exception as e:
                        logger.warning(f"Could not delete cached mp3 file {cached_mp3}: {e}")
                    st.session_state.converted_mp3_path = None
                    st.session_state.converted_mp3_file_key = None
                
                progress_bar.progress(1.0)
                status_text.text("✅ Transcription completed!")
                st.session_state.transcription = transcription
                st.session_state.filename = uploaded_file.name
                #st.success("🎉 Transcription completed successfully!")
            except Exception as e:
                st.error(f"❌ Error during transcription: {str(e)}")
                progress_bar.empty()
                status_text.empty()
    # Remove or comment out the following line to eliminate the empty frame/space:
    # st.markdown('<div style="margin-top:0.7rem;"></div>', unsafe_allow_html=True)
    # Show the bordered frame for transcription and controls only if there is a transcription
    if 'transcription' in st.session_state:
        
        transcription_text = st.text_area(
            "",
            value=st.session_state.transcription,
            height=120,
            help="The transcribed text from your audio file",
            label_visibility="visible"
        )
        st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            download_data = st.session_state.transcription.encode('utf-8')
            st.download_button(
                label="Download",
                data=download_data,
                file_name=f"{st.session_state.filename.split('.')[0]}_transcript.txt",
                mime="text/plain",
                help="Download the transcription as a text file"
            )
        with col2:
            if st.button("Copy to clipboard"):
                st.write("📋 Copied to clipboard!")
                st.code(st.session_state.transcription)
        st.markdown('</div>', unsafe_allow_html=True)
    # If no transcription, do not show the frame, placeholder, or empty text area

    # Model Selector (subtitle + select box, no bubble)
    # st.markdown('<div style="font-size:1.1rem;font-weight:600;margin-top:1.5rem;margin-bottom:0.5rem;">Model</div>', unsafe_allow_html=True)
    model = st.selectbox(
        "Transcription Model",
        ["Deepgram", "OpenAI Whisper"],
        key='model',
        help="Choose the transcription service to use. OpenAI Whisper often works better for non-English content."
    )

    # Other Info (polished, centered, no bubble)
    st.markdown('<hr style="margin:1.5rem 0 1rem 0;">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:1.1rem;font-weight:600;text-align:center;">About this app</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="font-size:1rem;line-height:1.6;text-align:center;">'
        'Supported formats: <b>MP3, WAV, M4A</b>.<br>'
        'Audio trimming and file chunking are automatic.<br>'
        'Created by <b>Juan Giraldo</b>.<br>'
        'Powered by <b>Streamlit</b>, <b>Deepgram</b>, and <b>OpenAI</b>.'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 