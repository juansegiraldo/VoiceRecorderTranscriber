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
import subprocess

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

def transcribe_with_openai(path: str, language: str = None) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
    with open(path, "rb") as audio_file:
        kwargs = {
            "model": "whisper-1",
            "file": audio_file
        }
        if language:
            kwargs["language"] = language
        transcript = client.audio.transcriptions.create(**kwargs)
    return transcript.text


def format_diarized_output(deepgram_response: dict, speaker_mapping: dict | None = None) -> str:
    """
    Format Deepgram response with utterances into a speaker-labeled transcript.
    Groups consecutive utterances from the same speaker together for better readability.
    
    Args:
        deepgram_response: Deepgram API JSON response with utterances
        speaker_mapping: Optional dict to remap speaker IDs (current_id -> new_id)
        
    Returns:
        Formatted string with speaker labels like "[Speaker 0]: text"
    """
    utterances = deepgram_response.get("results", {}).get("utterances", [])
    if not utterances:
        # Fallback to regular transcript if no utterances
        transcript = deepgram_response.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
        return transcript if transcript else ""
    
    formatted_lines = []
    current_speaker = None
    current_text_parts = []
    
    for utterance in utterances:
        speaker = utterance.get("speaker", 0)
        # Apply speaker mapping if provided
        if speaker_mapping and speaker in speaker_mapping:
            speaker = speaker_mapping[speaker]
        
        transcript = utterance.get("transcript", "").strip()
        
        if not transcript:
            continue
        
        # If same speaker, combine with previous text
        if speaker == current_speaker:
            current_text_parts.append(transcript)
        else:
            # New speaker - save previous speaker's combined text
            if current_speaker is not None and current_text_parts:
                combined_text = " ".join(current_text_parts)
                formatted_lines.append(f"[Speaker {current_speaker}]: {combined_text}")
            
            # Start new speaker
            current_speaker = speaker
            current_text_parts = [transcript]
    
    # Don't forget the last speaker
    if current_speaker is not None and current_text_parts:
        combined_text = " ".join(current_text_parts)
        formatted_lines.append(f"[Speaker {current_speaker}]: {combined_text}")
    
    return "\n".join(formatted_lines)


def extract_speakers_from_response(deepgram_response: dict) -> tuple[list[dict], dict]:
    """
    Extract utterances and speaker information from Deepgram response.
    
    Args:
        deepgram_response: Deepgram API JSON response with utterances
        
    Returns:
        tuple: (list of utterances with speaker info, speaker_stats dict)
    """
    utterances = deepgram_response.get("results", {}).get("utterances", [])
    if not utterances:
        return [], {}
    
    # Count speaker occurrences
    speaker_stats = {}
    for utterance in utterances:
        speaker = utterance.get("speaker", 0)
        speaker_stats[speaker] = speaker_stats.get(speaker, 0) + 1
    
    return utterances, speaker_stats


def map_speakers_between_chunks(prev_speakers: dict, current_speakers: dict, 
                                 prev_last_speaker: int | None = None) -> dict:
    """
    Map speakers from current chunk to previous chunk speakers.
    
    Args:
        prev_speakers: Speaker stats from previous chunk {speaker_id: count}
        current_speakers: Speaker stats from current chunk {speaker_id: count}
        prev_last_speaker: Last speaker ID from previous chunk
        
    Returns:
        dict: Mapping from current speaker ID to previous speaker ID
    """
    if not prev_speakers or not current_speakers:
        return {}
    
    mapping = {}
    
    # Strategy 1: If there's a clear last speaker from previous chunk,
    # try to match it with the first speaker of current chunk
    if prev_last_speaker is not None:
        # Find the most common speaker in current chunk (likely the first one)
        most_common_current = max(current_speakers.items(), key=lambda x: x[1])[0]
        # If previous chunk ended with a speaker, try to match it
        if prev_last_speaker in prev_speakers:
            mapping[most_common_current] = prev_last_speaker
    
    # Strategy 2: Map speakers by frequency/order
    # Sort speakers by frequency (most common first)
    prev_sorted = sorted(prev_speakers.items(), key=lambda x: x[1], reverse=True)
    current_sorted = sorted(current_speakers.items(), key=lambda x: x[1], reverse=True)
    
    # Map by order (most common to most common)
    for i, (current_speaker, _) in enumerate(current_sorted):
        if current_speaker not in mapping:  # Don't override existing mapping
            if i < len(prev_sorted):
                # Map to corresponding speaker from previous chunk
                mapping[current_speaker] = prev_sorted[i][0]
            else:
                # New speaker not seen before - assign new ID
                max_prev_speaker = max(prev_speakers.keys()) if prev_speakers else -1
                mapping[current_speaker] = max_prev_speaker + 1
    
    return mapping


def transcribe_with_deepgram(path: str, language: str = None, diarize: bool = False, return_raw: bool = False) -> str | dict:
    """
    Transcribe audio using Deepgram API.
    
    Args:
        path: Path to audio file
        language: Language code ('es' or 'en')
        diarize: Whether to enable speaker diarization
        return_raw: If True, return raw response dict instead of formatted string
        
    Returns:
        Formatted transcript string or raw response dict if return_raw=True
    """
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPGRAM_API_KEY environment variable not set")
    # Determinar el código de idioma para Deepgram
    lang_code = "es" if language == "es" else "en"
    
    # Build URL parameters
    base_params = "smart_format=true&punctuate=true"
    if diarize:
        base_params += "&diarize=true&utterances=true"
    else:
        base_params += "&diarize=false"
    
    url_default = f"https://api.deepgram.com/v1/listen?{base_params}&language={lang_code}&model=base"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/mp3",
        "Accept": "application/json",
    }
    with open(path, "rb") as f:
        audio_data = f.read()
    response = requests.post(url_default, headers=headers, data=audio_data)
    if not response.ok:
        print(f"Deepgram error response ({lang_code} model):", response.text)
        if return_raw:
            return {"error": response.text}
        return f"Deepgram error ({lang_code} model): {response.text}"
    dg = response.json()
    print(f"Deepgram raw response ({lang_code} model):", dg)
    
    # If return_raw is requested, return the raw response
    if return_raw:
        return dg
    
    # Handle diarized response
    if diarize:
        formatted = format_diarized_output(dg)
        if formatted and len(formatted.strip()) > 10:
            return formatted
    else:
        transcript = dg.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
        if transcript and len(transcript.strip()) > 10:
            return transcript
    
    # Si no hay buen resultado, intentar autodetección
    url_auto = f"https://api.deepgram.com/v1/listen?{base_params}&detect_language=true"
    response2 = requests.post(url_auto, headers=headers, data=audio_data)
    if not response2.ok:
        print("Deepgram error response (auto-detect):", response2.text)
        if return_raw:
            return {"error": response2.text}
        return f"Deepgram error (auto-detect): {response2.text}"
    dg2 = response2.json()
    print("Deepgram raw response (auto-detect):", dg2)
    
    # Handle diarized response
    if diarize:
        formatted2 = format_diarized_output(dg2)
        if formatted2 and len(formatted2.strip()) > 10:
            return formatted2
    else:
        transcript2 = dg2.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
        if transcript2 and len(transcript2.strip()) > 10:
            return transcript2
    
    # Si todo falla, intentar modelo original
    url_original = f"https://api.deepgram.com/v1/listen?{base_params}"
    response3 = requests.post(url_original, headers=headers, data=audio_data)
    if not response3.ok:
        print("Deepgram error response (original):", response3.text)
        if return_raw:
            return {"error": response3.text}
        return f"Deepgram error (original): {response3.text}"
    dg3 = response3.json()
    print("Deepgram raw response (original):", dg3)
    
    # Handle diarized response
    if diarize:
        formatted3 = format_diarized_output(dg3)
        if formatted3:
            return formatted3
    else:
        transcript3 = dg3.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
        if transcript3:
            return transcript3
    
    if return_raw:
        return {"error": "All attempts failed"}
    return f"Deepgram failed to transcribe properly. Responses:\n{lang_code}: {dg}\nAuto-detect: {dg2}\nOriginal: {dg3}"


def transcribe_file(path: str, model: str, language: str = None, diarize: bool = False) -> str:
    if model == "OpenAI Whisper":
        return transcribe_with_openai(path, language)
    elif model == "Deepgram":
        return transcribe_with_deepgram(path, language, diarize)
    else:
        raise ValueError(f"Unknown model: {model}")


def transcribe_large_file_with_diarization(chunk_paths: list[str], language: str = None, progress_bar=None, status_text=None) -> str:
    """
    Transcribe large file with speaker diarization, maintaining speaker consistency across chunks.
    
    Args:
        chunk_paths: List of chunk file paths
        language: Language code ('es' or 'en')
        progress_bar: Streamlit progress bar
        status_text: Streamlit status text element
        
    Returns:
        Combined transcription with consistent speaker IDs across chunks
    """
    transcriptions = []
    temp_dir = os.path.dirname(chunk_paths[0]) if len(chunk_paths) > 1 else None
    
    # Track speakers across chunks
    global_speaker_mapping = {}  # Maps (chunk_idx, local_speaker) -> global_speaker
    prev_speakers = {}
    prev_last_speaker = None
    next_global_speaker_id = 0
    
    try:
        for chunk_idx, chunk_path in enumerate(chunk_paths):
            chunk_num = chunk_idx + 1
            if status_text:
                status_text.text(f"🎤 Processing chunk {chunk_num}/{len(chunk_paths)}...")
            if progress_bar:
                progress_bar.progress(chunk_num / len(chunk_paths))
            
            try:
                # Get raw response for this chunk
                raw_response = transcribe_with_deepgram(chunk_path, language, diarize=True, return_raw=True)
                
                if isinstance(raw_response, dict) and "error" in raw_response:
                    transcriptions.append(f"[Error in chunk {chunk_num}: {raw_response['error']}]")
                    continue
                
                # Extract speaker information
                utterances, current_speakers = extract_speakers_from_response(raw_response)
                
                if not utterances:
                    # Fallback to regular transcription
                    chunk_transcription = transcribe_with_deepgram(chunk_path, language, diarize=False)
                    transcriptions.append(chunk_transcription)
                    continue
                
                # Map speakers to maintain consistency
                speaker_mapping = {}
                if chunk_idx > 0 and prev_speakers:
                    # Map current chunk speakers to previous chunk speakers
                    local_mapping = map_speakers_between_chunks(
                        prev_speakers, 
                        current_speakers, 
                        prev_last_speaker
                    )
                    
                    # Convert local mapping to global speaker IDs
                    for local_speaker, mapped_speaker in local_mapping.items():
                        # Find the global ID for the mapped speaker from previous chunk
                        prev_global_id = None
                        for (prev_chunk_idx, prev_local_speaker), global_id in global_speaker_mapping.items():
                            if prev_chunk_idx == chunk_idx - 1 and prev_local_speaker == mapped_speaker:
                                prev_global_id = global_id
                                break
                        
                        if prev_global_id is not None:
                            speaker_mapping[local_speaker] = prev_global_id
                        else:
                            # New speaker
                            speaker_mapping[local_speaker] = next_global_speaker_id
                            next_global_speaker_id += 1
                else:
                    # First chunk - assign global IDs sequentially
                    for local_speaker in current_speakers.keys():
                        speaker_mapping[local_speaker] = next_global_speaker_id
                        next_global_speaker_id += 1
                
                # Store mapping for future chunks
                for local_speaker, global_speaker in speaker_mapping.items():
                    global_speaker_mapping[(chunk_idx, local_speaker)] = global_speaker
                
                # Format with mapped speakers
                chunk_transcription = format_diarized_output(raw_response, speaker_mapping)
                transcriptions.append(chunk_transcription)
                
                # Update tracking for next chunk
                prev_speakers = current_speakers
                if utterances:
                    # Keep the local speaker ID (not global) for mapping in next chunk
                    prev_last_speaker = utterances[-1].get("speaker", 0)
                
                if status_text:
                    status_text.text(f"✅ Chunk {chunk_num} processed")
                    
            except Exception as e:
                if status_text:
                    status_text.text(f"❌ Error processing chunk {chunk_num}: {str(e)}")
                transcriptions.append(f"[Error in chunk {chunk_num}: {str(e)}]")
    
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Join transcriptions with spacing
    combined_transcription = "\n\n".join(transcriptions)
    return combined_transcription


def transcribe_large_file(file_path: str, model: str, language: str = None, diarize: bool = False, progress_bar=None, status_text=None) -> str:
    if status_text:
        status_text.text("🔍 Analyzing file...")
    chunk_paths = split_audio_file(file_path)
    if len(chunk_paths) == 1:
        if status_text:
            status_text.text("🎤 Transcribing file...")
        return transcribe_file(file_path, model, language, diarize)
    if status_text:
        status_text.text(f"📦 File split into {len(chunk_paths)} chunks")
    
    # Special handling for diarized multi-chunk files
    if diarize and model == "Deepgram":
        return transcribe_large_file_with_diarization(chunk_paths, language, progress_bar, status_text)
    
    # Regular processing for non-diarized or non-Deepgram
    transcriptions = []
    temp_dir = os.path.dirname(chunk_paths[0]) if len(chunk_paths) > 1 else None
    try:
        for i, chunk_path in enumerate(chunk_paths, 1):
            if status_text:
                status_text.text(f"🎤 Processing chunk {i}/{len(chunk_paths)}...")
            if progress_bar:
                progress_bar.progress(i / len(chunk_paths))
            try:
                chunk_transcription = transcribe_file(chunk_path, model, language, diarize)
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
    # Join transcriptions - use newlines for diarized output to preserve speaker labels
    if diarize:
        combined_transcription = "\n\n".join(transcriptions)
    else:
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
        allowed_types.append('mp4')
    uploaded_file = st.file_uploader(
        "",
        type=allowed_types,
        help="Select an audio file to transcribe (MP3, WAV, M4A, MP4)" + (" (M4A/MP4 extraction requires FFmpeg)" if not ffmpeg_available else ""),
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
        
        # Check if file actually changed
        current_file_key = st.session_state.get('current_file_key')
        file_key_changed = current_file_key != file_key
        
        # For M4A/MP4 files, check if we have a valid cached conversion
        has_valid_cache = False
        if file_extension in ['m4a', 'mp4']:
            cached_path = st.session_state.get('converted_mp3_path')
            cached_key = st.session_state.get('converted_mp3_file_key')
            # Valid cache if: same file key AND file exists
            if cached_path and cached_key == file_key and os.path.exists(cached_path):
                has_valid_cache = True
        
        # File changed if: different file key OR (M4A/MP4 without valid cache)
        file_changed = file_key_changed or (file_extension in ['m4a', 'mp4'] and not has_valid_cache)
        
        # Clean up old conversion if file changed
        if file_key_changed and st.session_state.get('converted_mp3_path'):
            try:
                old_cached_path = st.session_state.converted_mp3_path
                if old_cached_path and isinstance(old_cached_path, str) and os.path.exists(old_cached_path):
                    os.unlink(old_cached_path)
            except Exception as e:
                logger.warning(f"Could not delete old cached mp3 file {old_cached_path}: {e}")
            st.session_state.converted_mp3_path = None
            st.session_state.converted_mp3_file_key = None
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
                elif file_extension == 'mp4':
                    with st.spinner("🔄 Extracting audio from MP4..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_tmp_file:
                            mp3_file_path = mp3_tmp_file.name
                        # Use ffmpeg to extract audio from MP4
                        # This requires ffmpeg to be installed and in PATH
                        if not ffmpeg_available:
                            raise Exception("MP4 extraction requires FFmpeg. Please ensure it's installed and in your PATH.")
                        cmd = f"ffmpeg -i {tmp_file_path} -vn -acodec libmp3lame -ab 192k -ar 44100 -y {mp3_file_path}"
                        try:
                            subprocess.run(cmd, shell=True, check=True)
                            audio_path = mp3_file_path
                            st.session_state.converted_mp3_path = audio_path
                            st.session_state.converted_mp3_file_key = file_key
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Error extracting audio from MP4: {e}")
                            raise Exception(f"Error extracting audio from MP4: {e}")
                        except Exception as e:
                            logger.error(f"Error during MP4 extraction: {e}")
                            raise e
                else:
                    audio_path = tmp_file_path
                    st.session_state.converted_mp3_path = None
                    st.session_state.converted_mp3_file_key = None
                audio = AudioSegment.from_file(audio_path)
                duration_ms = len(audio)
                duration_seconds = duration_ms / 1000
                if file_extension != 'm4a' and file_extension != 'mp4':
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
            # Obtener idioma seleccionado
            language_ui = st.session_state.get('language', '🇪🇸 Español')
            # Mapear a código de idioma
            language_code = 'es' if 'es' in language_ui.lower() else 'en'
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
                # Use cached MP3 if available (to avoid re-conversion)
                audio_path: str = tmp_file_path  # Default
                use_cached_mp3 = False
                if file_extension == 'm4a' and st.session_state.get('converted_mp3_path') and st.session_state.get('converted_mp3_file_key') == file_key:
                    if st.session_state.converted_mp3_path is not None:
                        audio_path = st.session_state.converted_mp3_path
                        use_cached_mp3 = True
                elif file_extension == 'mp4' and st.session_state.get('converted_mp3_path') and st.session_state.get('converted_mp3_file_key') == file_key:
                    if st.session_state.converted_mp3_path is not None:
                        audio_path = st.session_state.converted_mp3_path
                        use_cached_mp3 = True
                
                # Only convert if we don't have a cached MP3
                if not use_cached_mp3:
                    if file_extension == 'm4a':
                        if status_text:
                            status_text.text("🔄 Converting M4A to MP3...")
                        time.sleep(0.5)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_tmp_file:
                            mp3_file_path = mp3_tmp_file.name
                        converted_path = convert_m4a_to_mp3(tmp_file_path, mp3_file_path)
                        temp_files_to_cleanup.append(converted_path)
                        audio_path = converted_path
                    elif file_extension == 'mp4':
                        if status_text:
                            status_text.text("🔄 Extracting audio from MP4...")
                        time.sleep(0.5)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_tmp_file:
                            mp3_file_path = mp3_tmp_file.name
                        # Use ffmpeg to extract audio from MP4
                        # This requires ffmpeg to be installed and in PATH
                        if not ffmpeg_available:
                            raise Exception("MP4 extraction requires FFmpeg. Please ensure it's installed and in your PATH.")
                        cmd = f"ffmpeg -i {tmp_file_path} -vn -acodec libmp3lame -ab 192k -ar 44100 -y {mp3_file_path}"
                        try:
                            subprocess.run(cmd, shell=True, check=True)
                            audio_path = mp3_file_path
                            temp_files_to_cleanup.append(audio_path)
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Error extracting audio from MP4: {e}")
                            raise Exception(f"Error extracting audio from MP4: {e}")
                        except Exception as e:
                            logger.error(f"Error during MP4 extraction: {e}")
                            raise e
                
                # Apply trimming if settings are provided
                # Note: audio_path is already MP3 at this point (either from cache or conversion)
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
                    # Get diarize setting (only use if Deepgram is selected)
                    model = st.session_state.get('model', 'Deepgram')
                    diarize_setting = st.session_state.get('diarize', False) if model == "Deepgram" else False
                    transcription = transcribe_large_file(
                        audio_path, 
                        model,
                        language=language_code,
                        diarize=diarize_setting,
                        progress_bar=progress_bar, 
                        status_text=status_text,
                    )
                else:
                    st.error("❌ Internal error: audio_path is None.")
                    return
                # Clean up all temp files (but keep cached mp3 for potential reuse)
                for temp_file in temp_files_to_cleanup:
                    try:
                        # Don't delete the cached mp3 if it's in the cleanup list - we want to keep it
                        if temp_file and isinstance(temp_file, str) and os.path.exists(temp_file):
                            # Only delete if it's not the cached mp3 we want to preserve
                            cached_mp3 = st.session_state.get('converted_mp3_path')
                            if temp_file != cached_mp3:
                                os.unlink(temp_file)
                    except Exception as e:
                        logger.warning(f"Could not delete temp file {temp_file}: {e}")
                # Keep converted_mp3_path and converted_mp3_file_key in session state
                # They will be cleaned up automatically when a new file is uploaded
                
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
        
        # Show transcription in a code block with copy button
        st.code(st.session_state.transcription, language=None)
        
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
            # Remove the copy button since st.code() has built-in copy functionality
            st.markdown('<div style="height:2.5rem;"></div>', unsafe_allow_html=True)  # Spacer to align with download button
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

    # Language Selector (Español/Inglés)
    language = st.selectbox(
        "Language",
        ["🇪🇸 Español", "🇬🇧 English"],
        key='language',
        help="Select the language of the audio for better transcription accuracy."
    )

    # Speaker Diarization Toggle (only for Deepgram)
    diarize_enabled = st.checkbox(
        "Identify different speakers (diarization)",
        key='diarize',
        value=True,  # Default to enabled
        disabled=(model == "OpenAI Whisper"),
        help="Enable speaker identification to show who said what. Only works with Deepgram model."
    )

    # Other Info (polished, centered, no bubble)
    st.markdown('<hr style="margin:1.5rem 0 1rem 0;">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:1.1rem;font-weight:600;text-align:center;">About this app</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="font-size:1rem;line-height:1.6;text-align:center;">'
        'Supported formats: <b>MP3, WAV, M4A, MP4</b>.<br>'
        '<a href="https://chatgpt.com/g/g-6874e87c48608191afa2da8e3e769279-generadoractasreunion" target="_blank">Usa este GPT para generar el acta</a><br>'
        'Created by <b>Juan Giraldo</b>.<br>'
        'Powered by <b>Streamlit</b>, <b>Deepgram</b>, and <b>OpenAI</b>.'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 
