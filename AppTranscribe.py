import streamlit as st
import os
from openai import OpenAI
import requests
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import json
import base64
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


def get_secret(key, default=None):
    """Read a secret preferring Streamlit Cloud's st.secrets, falling back to
    environment variables (loaded from .env locally via python-dotenv).

    st.secrets raises if no secrets.toml exists (e.g. local dev), so we guard it.
    """
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


# Server-side safety net for finished transcripts. Mobile browsers suspend
# background tabs, which drops the Streamlit session and discards
# st.session_state — losing the result of a long transcription right as it
# finishes. Persisting to the container's temp dir lets a fresh session
# (page refresh) recover the last completed transcript.
_LAST_TRANSCRIPT_PATH = os.path.join(tempfile.gettempdir(), "vt_last_transcript.json")


def _save_last_transcript(filename: str, transcription: str) -> None:
    try:
        with open(_LAST_TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "filename": filename,
                "transcription": transcription,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f)
    except Exception as e:
        logger.warning(f"Could not persist last transcript: {e}")


def _load_last_transcript() -> dict | None:
    try:
        if os.path.exists(_LAST_TRANSCRIPT_PATH):
            with open(_LAST_TRANSCRIPT_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("transcription"):
                return data
    except Exception as e:
        logger.warning(f"Could not load last transcript: {e}")
    return None


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
    layout="centered",
    initial_sidebar_state="collapsed"
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

# Non-diarized fallback chain, best model first (mirrors the Deepgram 3-attempt chain)
OPENAI_TRANSCRIBE_MODELS = ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"]
OPENAI_DIARIZE_MODEL = "gpt-4o-transcribe-diarize"


def _get_openai_client() -> OpenAI:
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, timeout=300.0, max_retries=2)


def transcribe_openai_diarized_raw(client: OpenAI, path: str, language: str = None,
                                   known_speaker_names: list[str] | None = None,
                                   known_speaker_clips: list[bytes] | None = None) -> dict:
    """
    Single call to OpenAI's diarization model (gpt-4o-transcribe-diarize).

    Returns the parsed diarized_json dict, or {"error": ...} on failure — the
    same contract as transcribe_with_deepgram(return_raw=True).

    known_speaker_names/known_speaker_clips (max 4; clips must be 2-10s of a
    single speaker) anchor speaker identity across chunked calls: segments the
    API matches to a reference come back labeled with that name instead of a
    generic "A"/"B". Sent via extra_body so older SDK versions without the typed
    kwargs still work.
    """
    with open(path, "rb") as f:
        audio_bytes = f.read()
    kwargs = {
        "model": OPENAI_DIARIZE_MODEL,
        "response_format": "diarized_json",
        # Required by the API for audio longer than 30s; harmless below that.
        "chunking_strategy": "auto",
    }
    if language:
        kwargs["language"] = language
    extra_body = None
    if known_speaker_names and known_speaker_clips:
        extra_body = {
            "known_speaker_names": known_speaker_names,
            "known_speaker_references": [
                "data:audio/mp3;base64," + base64.b64encode(clip).decode("ascii")
                for clip in known_speaker_clips
            ],
        }
    file_arg = (os.path.basename(path), audio_bytes)
    try:
        response = client.audio.transcriptions.with_raw_response.create(
            file=file_arg, extra_body=extra_body, **kwargs
        )
        return json.loads(response.text)
    except Exception as e:
        # language is best-effort on the diarize model; retry once without it
        if language and "language" in str(e).lower():
            try:
                kwargs.pop("language", None)
                response = client.audio.transcriptions.with_raw_response.create(
                    file=file_arg, extra_body=extra_body, **kwargs
                )
                return json.loads(response.text)
            except Exception as e2:
                print("OpenAI diarize error (no-language retry):", e2)
                return {"error": str(e2)}
        print("OpenAI diarize error:", e)
        return {"error": str(e)}


def openai_segments_to_deepgram_shape(openai_response: dict, label_to_speaker: dict | None = None) -> tuple[dict, dict]:
    """
    Adapt OpenAI diarized_json (segments with string speaker labels) into the
    Deepgram-shaped dict the existing diarization helpers consume:
    {"results": {"utterances": [{"speaker": int, "transcript": str, ...}]}}.

    label_to_speaker may be pre-seeded (e.g. {"S3": 3}) so segments matched to a
    known-speaker reference land directly on their global integer ID; unseen
    labels get the next free integer in order of first appearance.

    Returns (shaped_dict, final label->int mapping).
    """
    label_to_speaker = dict(label_to_speaker or {})
    utterances = []
    for segment in openai_response.get("segments", []):
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        label = str(segment.get("speaker", "A"))
        if label not in label_to_speaker:
            label_to_speaker[label] = max(label_to_speaker.values(), default=-1) + 1
        utterances.append({
            "speaker": label_to_speaker[label],
            "transcript": text,
            "start": float(segment.get("start") or 0.0),
            "end": float(segment.get("end") or 0.0),
        })
    return {"results": {"utterances": utterances}}, label_to_speaker


def extract_speaker_reference_clip(chunk_audio: AudioSegment, utterances: list[dict],
                                   speaker_id: int, out_dir: str) -> str | None:
    """
    Cut a 2-10s single-speaker MP3 clip to use as a known_speaker_reference in
    later chunks. Utterance timestamps are relative to the chunk the audio came
    from, so no offset math is needed. Returns the clip path, or None if the
    speaker has no utterance long enough (or extraction fails) — never raises.
    """
    try:
        MARGIN_MS = 150   # shave edges: segment timestamps can be slightly loose
        MIN_MS = 2500     # 0.5s safety margin over the API's 2s floor
        MAX_MS = 8000     # comfortably under the API's 10s cap
        best = None
        best_len = 0
        for utterance in utterances:
            if utterance.get("speaker") != speaker_id:
                continue
            start_ms = int(utterance.get("start", 0) * 1000) + MARGIN_MS
            end_ms = int(utterance.get("end", 0) * 1000) - MARGIN_MS
            if end_ms - start_ms > best_len:
                best_len = end_ms - start_ms
                best = (start_ms, end_ms)
        if best is None or best_len < MIN_MS:
            return None
        start_ms, end_ms = best
        if best_len > MAX_MS:
            # Take the middle of the utterance — least likely to bleed into a neighbor
            mid = (start_ms + end_ms) // 2
            start_ms, end_ms = mid - MAX_MS // 2, mid + MAX_MS // 2
        clip_path = os.path.join(out_dir, f"speaker_ref_{speaker_id}.mp3")
        chunk_audio[start_ms:end_ms].export(clip_path, format="mp3")
        return clip_path
    except Exception as e:
        print(f"Could not extract reference clip for speaker {speaker_id}:", e)
        return None


def transcribe_with_openai(path: str, language: str = None, diarize: bool = False, return_raw: bool = False) -> str | dict:
    """
    Transcribe audio using OpenAI.

    Non-diarized runs walk a fallback chain of models (best first, accept only
    transcripts >10 chars — mirrors the Deepgram chain). Diarized runs use
    gpt-4o-transcribe-diarize and fall back to plain transcription if it yields
    nothing usable.

    Returns a formatted transcript string, or the raw response dict
    ({"error": ...} on failure) when return_raw=True.
    """
    client = _get_openai_client()

    if diarize:
        raw = transcribe_openai_diarized_raw(client, path, language)
        if return_raw:
            return raw
        if not (isinstance(raw, dict) and "error" in raw):
            shaped, _ = openai_segments_to_deepgram_shape(raw)
            formatted = format_diarized_output(shaped)
            if formatted and len(formatted.strip()) > 10:
                return formatted
        # Diarization produced nothing usable — fall through to the plain chain

    last_error = None
    for model_name in OPENAI_TRANSCRIBE_MODELS:
        try:
            with open(path, "rb") as audio_file:
                kwargs = {
                    "model": model_name,
                    "file": audio_file,
                    "response_format": "text",
                }
                if language:
                    kwargs["language"] = language
                result = client.audio.transcriptions.create(**kwargs)
            text = result if isinstance(result, str) else getattr(result, "text", "")
            text = (text or "").strip()
            if len(text) > 10:
                return text
            if text and model_name == OPENAI_TRANSCRIBE_MODELS[-1]:
                return text
            print(f"OpenAI {model_name} returned too little text, trying next model")
        except Exception as e:
            last_error = e
            print(f"OpenAI error ({model_name}):", e)
    return f"OpenAI failed to transcribe properly. Last error: {last_error}"


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
    api_key = get_secret("DEEPGRAM_API_KEY")
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
    if model == "OpenAI":
        return transcribe_with_openai(path, language, diarize)
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


def transcribe_large_file_with_diarization_openai(chunk_paths: list[str], language: str = None, progress_bar=None, status_text=None) -> str:
    """
    OpenAI counterpart of transcribe_large_file_with_diarization.

    gpt-4o-transcribe-diarize speaker labels are only consistent within one API
    call, so consistency across chunks is kept two ways (hybrid):
      1. Native anchoring: up to 4 reference clips of already-identified
         speakers are sent as known_speaker_references, so the API labels
         matching segments "S<global_id>" directly.
      2. Heuristic fallback: labels the API didn't match to a reference are
         resolved with the existing map_speakers_between_chunks
         frequency/last-speaker logic.
    """
    transcriptions = []
    temp_dir = os.path.dirname(chunk_paths[0]) if len(chunk_paths) > 1 else None
    client = _get_openai_client()

    speaker_clips = {}     # global_id -> reference clip path
    global_counts = {}     # global_id -> cumulative utterance count (picks top-4 refs)
    prev_stats = {}        # previous chunk's utterance counts keyed by global id
    prev_last_speaker = None
    next_global_id = 0

    try:
        for chunk_idx, chunk_path in enumerate(chunk_paths):
            chunk_num = chunk_idx + 1
            if status_text:
                status_text.text(f"🎤 Processing chunk {chunk_num}/{len(chunk_paths)} (matching speakers)...")
            if progress_bar:
                progress_bar.progress(chunk_num / len(chunk_paths))

            try:
                # Known-speaker refs: the 4 most-talkative identified speakers
                ref_gids = sorted(speaker_clips.keys(), key=lambda g: global_counts.get(g, 0), reverse=True)[:4]
                names, clips = [], []
                for gid in ref_gids:
                    try:
                        with open(speaker_clips[gid], "rb") as f:
                            clips.append(f.read())
                        names.append(f"S{gid}")
                    except Exception:
                        pass
                ref_gids = [int(name[1:]) for name in names]  # stay aligned if a clip read failed

                raw = transcribe_openai_diarized_raw(client, chunk_path, language, names or None, clips or None)

                if isinstance(raw, dict) and "error" in raw:
                    # Whole-call failure: degrade to plain transcription for this chunk
                    transcriptions.append(transcribe_with_openai(chunk_path, language, diarize=False))
                    continue

                seed = {f"S{gid}": gid for gid in ref_gids}
                shaped, label_map = openai_segments_to_deepgram_shape(raw, seed)
                utterances = shaped["results"]["utterances"]
                if not utterances:
                    # Silent/empty chunk as far as diarization goes
                    transcriptions.append(transcribe_with_openai(chunk_path, language, diarize=False))
                    continue

                # Resolve labels the API didn't match to a reference
                unmatched_ids = {pid for label, pid in label_map.items() if label not in seed}
                remap = {}
                if unmatched_ids:
                    cur_stats = {}
                    for utterance in utterances:
                        if utterance["speaker"] in unmatched_ids:
                            cur_stats[utterance["speaker"]] = cur_stats.get(utterance["speaker"], 0) + 1
                    uncovered_prev = {g: c for g, c in prev_stats.items() if g not in ref_gids}
                    if not uncovered_prev:
                        # First chunk, or every prior speaker is covered by a ref:
                        # unmatched labels are genuinely new speakers
                        for pid in cur_stats:  # insertion order = first appearance
                            remap[pid] = next_global_id
                            next_global_id += 1
                    else:
                        heuristic = map_speakers_between_chunks(
                            uncovered_prev,
                            cur_stats,
                            prev_last_speaker if prev_last_speaker in uncovered_prev else None,
                        )
                        for pid, gid in heuristic.items():
                            if gid in uncovered_prev:
                                remap[pid] = gid
                            else:
                                # Heuristic's "new speaker" arithmetic can collide
                                # with a ref-covered gid — use the registry counter
                                remap[pid] = next_global_id
                                next_global_id += 1
                    for utterance in utterances:
                        utterance["speaker"] = remap.get(utterance["speaker"], utterance["speaker"])

                # Speakers already carry global IDs — no mapping arg needed
                transcriptions.append(format_diarized_output(shaped))

                # Bookkeeping for the next chunk
                chunk_stats = {}
                for utterance in utterances:
                    chunk_stats[utterance["speaker"]] = chunk_stats.get(utterance["speaker"], 0) + 1
                for gid, count in chunk_stats.items():
                    global_counts[gid] = global_counts.get(gid, 0) + count
                prev_stats = chunk_stats
                prev_last_speaker = utterances[-1]["speaker"]
                next_global_id = max(next_global_id, max(chunk_stats) + 1)

                # Harvest reference clips for speakers that don't have one yet
                if temp_dir:
                    missing = [gid for gid in chunk_stats if gid not in speaker_clips]
                    if missing:
                        chunk_audio = AudioSegment.from_file(chunk_path)
                        for gid in missing:
                            clip_path = extract_speaker_reference_clip(chunk_audio, utterances, gid, temp_dir)
                            if clip_path:
                                speaker_clips[gid] = clip_path

                if status_text:
                    status_text.text(f"✅ Chunk {chunk_num} processed")

            except Exception as e:
                if status_text:
                    status_text.text(f"❌ Error processing chunk {chunk_num}: {str(e)}")
                transcriptions.append(f"[Error in chunk {chunk_num}: {str(e)}]")

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return "\n\n".join(transcriptions)


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
    if diarize and model == "OpenAI":
        return transcribe_large_file_with_diarization_openai(chunk_paths, language, progress_bar, status_text)

    # Regular processing for non-diarized
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


# ---------------------------------------------------------------------------
# Google Drive source
#
# A DriveFile mimics the subset of Streamlit's UploadedFile interface that the
# transcription pipeline relies on (.name / .size / .type / .getvalue()), so a
# file fetched from Drive is indistinguishable from a local upload to all code
# downstream of the uploader. This means the conversion cache (keyed on
# name+size+type), M4A/MP4 conversion, duration analysis, trimming and
# transcription all work unchanged.
# ---------------------------------------------------------------------------

# Scope: read-only access is enough to list and download the user's audio.
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
# Extensions the app can transcribe; used as a safety net when a Drive file's
# mimeType is generic (e.g. application/octet-stream).
DRIVE_AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.mp4')


class DriveFile:
    """Duck-typed stand-in for st.runtime.uploaded_file_manager.UploadedFile."""

    def __init__(self, name, data_bytes, mime_type, file_id):
        self.name = name  # MUST keep a real extension: downstream routes by name.split('.')[-1]
        self._data = data_bytes
        self.size = len(data_bytes)
        # fileId is folded into .type only to keep file_key unique across two
        # Drive files that share name+size. .type is never parsed downstream.
        self.type = f"{mime_type}|{file_id}"

    def getvalue(self):
        return self._data


def _drive_configured():
    """True only if the three Google OAuth secrets are present."""
    return all(get_secret(k) for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REDIRECT_URI"))


# --- PKCE hand-off across the OAuth redirect ------------------------------
# When Google redirects back to the app, the browser opens a BRAND-NEW
# Streamlit session, so st.session_state from the login click is gone. The
# `state` value does survive — it round-trips through the URL. So we stash the
# PKCE code_verifier on disk keyed by `state`, and the callback (which reads
# `state` from the URL) looks it up there. This is the standard workaround for
# OAuth-with-PKCE on Streamlit's per-session-memory model.
def _pkce_store_path():
    return os.path.join(tempfile.gettempdir(), "vt_oauth_pkce.json")


def _pkce_save(state, verifier):
    path = _pkce_store_path()
    try:
        store = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                store = json.load(f)
    except Exception:
        store = {}
    store[state] = verifier
    # Bound the file: keep only the last few pending logins.
    if len(store) > 10:
        store = dict(list(store.items())[-10:])
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f)
    except Exception as e:
        logger.warning(f"Could not persist PKCE verifier: {e}")


def _pkce_pop(state):
    path = _pkce_store_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            store = json.load(f)
    except Exception:
        return None
    verifier = store.pop(state, None)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f)
    except Exception:
        pass
    return verifier


def _build_drive_flow():
    """Create an OAuth Flow from secrets. Imported lazily so the app still runs
    if the google libraries aren't installed and Drive simply isn't used."""
    from google_auth_oauthlib.flow import Flow

    client_config = {
        "web": {
            "client_id": get_secret("GOOGLE_CLIENT_ID"),
            "client_secret": get_secret("GOOGLE_CLIENT_SECRET"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [get_secret("GOOGLE_REDIRECT_URI")],
        }
    }
    flow = Flow.from_client_config(client_config, scopes=DRIVE_SCOPES)
    flow.redirect_uri = get_secret("GOOGLE_REDIRECT_URI")
    return flow


def _credentials_to_dict(creds):
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }


def _get_drive_credentials():
    """Rebuild Credentials from the dict stashed in session_state, refreshing if
    expired. Returns None if the user isn't authenticated (or refresh fails)."""
    creds_dict = st.session_state.get("drive_creds")
    if not creds_dict:
        return None
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    creds = Credentials(**creds_dict)
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            st.session_state.drive_creds = _credentials_to_dict(creds)
        except Exception as e:
            logger.warning(f"Drive token refresh failed: {e}")
            # Force re-login by dropping the stale credentials.
            st.session_state.pop("drive_creds", None)
            return None
    return creds


def _handle_drive_oauth_callback():
    """If Google redirected back with ?code=..., exchange it for tokens once,
    then clear the query params (a spent code fails on rerun) and rerun."""
    params = st.query_params
    code = params.get("code")
    if not code or st.session_state.get("drive_creds"):
        return
    # The `state` from the URL is our key into the on-disk PKCE store. If it's
    # not there, this is a stale/foreign callback (also our CSRF guard: an
    # attacker cannot produce a `state` that indexes a verifier we saved).
    returned_state = params.get("state")
    verifier = _pkce_pop(returned_state) if returned_state else None
    if not verifier:
        st.error("❌ Google login expiró o no se pudo validar. Pulsa el botón de nuevo.")
        st.query_params.clear()
        return
    try:
        flow = _build_drive_flow()
        # Replay the PKCE code_verifier saved when we built the login URL;
        # without it Google rejects the exchange with "Missing code verifier".
        flow.code_verifier = verifier
        flow.fetch_token(code=code)
        st.session_state.drive_creds = _credentials_to_dict(flow.credentials)
    except Exception as e:
        logger.error(f"Drive token exchange failed: {e}")
        st.error(f"❌ Google login failed: {e}")
    finally:
        # Always clear the code so a rerun never re-uses it.
        st.query_params.clear()
    st.rerun()


def _get_drive_login_url():
    """Build the consent URL and remember the CSRF state + PKCE verifier."""
    flow = _build_drive_flow()
    auth_url, state = flow.authorization_url(
        access_type="offline",          # needed to receive a refresh_token
        include_granted_scopes="true",
        prompt="consent",               # force refresh_token on repeat logins
    )
    # PKCE: the code_verifier generated here must be replayed at token exchange,
    # which happens in a fresh Streamlit SESSION after the Google redirect — so
    # session_state won't carry it. Persist it on disk keyed by `state` (which
    # does survive, via the URL). The callback looks it up by the returned state.
    verifier = getattr(flow, "code_verifier", None)
    _pkce_save(state, verifier)
    return auth_url


def list_drive_audio(creds, page_size=25):
    """List the user's most recent audio/video files. Returns a list of dicts
    with id/name/size/mimeType."""
    from googleapiclient.discovery import build

    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    query = ("(mimeType contains 'audio/' or mimeType = 'video/mp4') "
             "and trashed = false")
    resp = service.files().list(
        q=query,
        orderBy="modifiedTime desc",
        pageSize=page_size,
        fields="files(id,name,size,mimeType,modifiedTime)",
        spaces="drive",
    ).execute()
    files = resp.get("files", [])
    # Safety net: also keep anything whose name has a supported extension, in
    # case the mimeType filter missed it (some m4a report odd mimetypes).
    seen = {f["id"] for f in files}
    if len(files) < page_size:
        resp2 = service.files().list(
            q="trashed = false",
            orderBy="modifiedTime desc",
            pageSize=page_size,
            fields="files(id,name,size,mimeType,modifiedTime)",
            spaces="drive",
        ).execute()
        for f in resp2.get("files", []):
            if f["id"] not in seen and f["name"].lower().endswith(DRIVE_AUDIO_EXTENSIONS):
                files.append(f)
    return files


def download_drive_file(creds, file_id, name, mime_type):
    """Download a Drive file into memory and wrap it as a DriveFile. Cached in
    session_state by file_id so slider reruns don't re-download."""
    cache = st.session_state.setdefault("drive_download_cache", {})
    if file_id in cache:
        return cache[file_id]

    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    drive_file = DriveFile(name, buffer.getvalue(), mime_type or "application/octet-stream", file_id)
    # Keep only the most recent download to bound memory.
    cache.clear()
    cache[file_id] = drive_file
    return drive_file


def render_drive_tab():
    """Render the Google Drive source tab. Returns a DriveFile if the user has
    loaded one, else None."""
    if not _drive_configured():
        st.info(
            "☁️ Google Drive no está configurado en este despliegue. "
            "Añade `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` y `GOOGLE_REDIRECT_URI` "
            "en los *secrets* para habilitarlo."
        )
        return None

    creds = _get_drive_credentials()
    if creds is None:
        st.markdown(
            '<div style="text-align:center;color:#9aa0a6;margin-bottom:0.8rem;">'
            'Inicia sesión con Google para elegir un audio de tu Drive.</div>',
            unsafe_allow_html=True,
        )
        try:
            login_url = _get_drive_login_url()
            st.link_button("🔑 Iniciar sesión con Google", login_url, use_container_width=True)
        except Exception as e:
            st.error(f"❌ No se pudo iniciar el login de Google: {e}")
        return None

    # Authenticated: list audio and let the user pick + load one.
    try:
        files = list_drive_audio(creds)
    except Exception as e:
        logger.error(f"Drive list failed: {e}")
        st.error(f"❌ No se pudieron listar los archivos de Drive: {e}")
        if st.button("Cerrar sesión de Google", key="drive_logout_err"):
            st.session_state.pop("drive_creds", None)
            st.rerun()
        return None

    if not files:
        st.info("No se encontraron archivos de audio recientes en tu Drive.")
        if st.button("Cerrar sesión de Google", key="drive_logout_empty"):
            st.session_state.pop("drive_creds", None)
            st.rerun()
        return None

    def _label(f):
        size_mb = f"{int(f['size']) / 1024 / 1024:.1f} MB" if f.get("size") else "?"
        modified = f.get("modifiedTime", "")[:10]
        return f"{f['name']} — {modified} — {size_mb}"

    options = {_label(f): f for f in files}
    choice = st.selectbox("Audios recientes en tu Drive", list(options.keys()), key="drive_file_choice")
    selected = options[choice]

    load_col, logout_col = st.columns([3, 1])
    loaded = None
    with load_col:
        if st.button("☁️ Cargar de Drive", type="primary", key="drive_load_btn", use_container_width=True):
            with st.spinner("Descargando de Google Drive..."):
                try:
                    loaded = download_drive_file(
                        creds, selected["id"], selected["name"], selected.get("mimeType")
                    )
                    st.session_state.drive_loaded_id = selected["id"]
                except Exception as e:
                    logger.error(f"Drive download failed: {e}")
                    st.error(f"❌ No se pudo descargar el archivo: {e}")
    with logout_col:
        if st.button("Salir", key="drive_logout", use_container_width=True):
            st.session_state.pop("drive_creds", None)
            st.session_state.pop("drive_download_cache", None)
            st.session_state.pop("drive_loaded_id", None)
            st.rerun()

    # On plain reruns (e.g. moving the trim slider), re-surface the already
    # downloaded file from cache instead of forcing another click.
    if loaded is None:
        cached_id = st.session_state.get("drive_loaded_id")
        cache = st.session_state.get("drive_download_cache", {})
        if cached_id and cached_id in cache:
            loaded = cache[cached_id]
    return loaded


def main():
    st.markdown(
        '''<style>
        html {
            -webkit-text-size-adjust: 100%; /* stop iOS from resizing text unexpectedly */
        }
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
        @media (max-width: 600px) {
            .block-container {
                max-width: 100% !important;
                padding: 0.5rem !important;
            }
        }
        /* --- Mobile-friendly tap targets --- */
        /* Full-width primary buttons and download button for easy thumb taps */
        .stButton > button,
        .stDownloadButton > button {
            width: 100%;
            min-height: 44px; /* Apple/Google recommended minimum touch target */
        }
        /* Comfortable height for select boxes and number inputs on touch */
        div[data-baseweb="select"] > div,
        .stNumberInput input,
        .stTextInput input {
            min-height: 44px;
            font-size: 16px !important; /* >=16px prevents iOS auto-zoom on focus */
        }
        /* File uploader: full width and a bit more breathing room */
        .stFileUploader {
            width: 100%;
        }
        </style>''', unsafe_allow_html=True)

    # Complete any pending Google Drive OAuth redirect before rendering the UI.
    _handle_drive_oauth_callback()

    # Logo and App Name (centered, no bubble)
    st.markdown('<div style="text-align:center;margin-bottom:1.2rem;"><span style="font-size:2.2rem;">🎤</span><div style="font-size:2rem;font-weight:700;margin-top:0.5rem;">Voice Transcriber</div></div>', unsafe_allow_html=True)

    # Transcription Section
    st.markdown('<div style="font-size:1.3rem;font-weight:600;margin-bottom:0.5rem;text-align:center;">Insert audio file</div>', unsafe_allow_html=True)
    allowed_types = ['mp3', 'wav']
    if ffmpeg_available:
        allowed_types.append('m4a')
        allowed_types.append('mp4')

    # Two input sources: a local upload or a file from the user's Google Drive.
    # Both converge on `uploaded_file` (a real UploadedFile or a DriveFile);
    # everything downstream is source-agnostic.
    tab_local, tab_drive = st.tabs(["📁 Subir archivo", "☁️ Google Drive"])
    with tab_local:
        # No `type=` filter on purpose: mobile pickers (notably Android/Chrome)
        # translate the extension filter into MIME types and gray out valid files
        # (.m4a is a known mismatch: audio/mp4 vs audio/x-m4a). Accept anything in
        # the picker and validate the extension server-side instead.
        local_file = st.file_uploader(
            "Audio file",
            help="Select an audio file to transcribe (MP3, WAV, M4A, MP4)" + (" (M4A/MP4 extraction requires FFmpeg)" if not ffmpeg_available else ""),
            label_visibility="collapsed"
        )
        if local_file is not None:
            _ext = local_file.name.split('.')[-1].lower()
            if _ext not in allowed_types:
                st.error(
                    f"❌ Formato no soportado: .{_ext}. "
                    f"Usa {', '.join(t.upper() for t in allowed_types)}."
                )
                local_file = None
    with tab_drive:
        drive_file = render_drive_tab()

    # Prefer whichever source the user last acted in; default to the local upload.
    if local_file is not None:
        st.session_state.last_input_source = "local"
    elif drive_file is not None and st.session_state.get("drive_loaded_id"):
        st.session_state.last_input_source = "drive"

    if st.session_state.get("last_input_source") == "drive":
        uploaded_file = drive_file if drive_file is not None else local_file
    else:
        uploaded_file = local_file if local_file is not None else drive_file
    
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
        st.markdown(f'<div style="font-size:0.9rem;color:#9aa0a6;text-align:center;margin-bottom:1rem;">🎵 Duration: {format_time(duration_seconds)}</div>', unsafe_allow_html=True)
        
        # Trimming controls
        st.markdown('<div style="font-size:1.1rem;font-weight:600;margin-bottom:0.5rem;text-align:center;">Audio Trimming (Optional)</div>', unsafe_allow_html=True)

        max_seconds = float(duration_seconds)

        # The slider and the two number inputs are three views of the same trim
        # range. To keep them in sync WITHOUT hitting Streamlit's "value ignored
        # because a key exists" trap, the widget keys themselves are the source of
        # truth: we seed them once per file, then each on_change callback writes the
        # reconciled values into the OTHER widgets' keys (callbacks run before the
        # widgets are re-instantiated, so this propagates cleanly). No `value=` args.
        if st.session_state.get('trim_state_key') != file_key:
            st.session_state.trim_range_slider = (0.0, max_seconds)
            st.session_state.trim_start_input = 0.0
            st.session_state.trim_end_input = max_seconds
            st.session_state.trim_state_key = file_key

        def _sync_from_slider():
            s, e = st.session_state.trim_range_slider
            st.session_state.trim_start_input = float(s)
            st.session_state.trim_end_input = float(e)

        def _sync_from_numbers():
            s = float(st.session_state.trim_start_input)
            e = float(st.session_state.trim_end_input)
            if s > e:  # keep start <= end
                s = e
                st.session_state.trim_start_input = s
            st.session_state.trim_range_slider = (s, e)

        # Coarse selection: two-handle range slider
        st.slider(
            "Select audio range to transcribe",
            min_value=0.0,
            max_value=max_seconds,
            step=0.1,
            format="%.1f s",
            help="Select the portion of the audio to transcribe (start and end times)",
            key="trim_range_slider",
            on_change=_sync_from_slider,
        )

        # Precise entry (much easier than dragging on a phone): numeric start/end
        num_col1, num_col2 = st.columns(2)
        with num_col1:
            st.number_input(
                "Start (s)",
                min_value=0.0,
                max_value=max_seconds,
                step=0.1,
                format="%.1f",
                key="trim_start_input",
                on_change=_sync_from_numbers,
            )
        with num_col2:
            st.number_input(
                "End (s)",
                min_value=0.0,
                max_value=max_seconds,
                step=0.1,
                format="%.1f",
                key="trim_end_input",
                on_change=_sync_from_numbers,
            )

        start_time, end_time = st.session_state.trim_range_slider

        # Show trim preview
        trim_duration = end_time - start_time
        st.markdown(f'<div style="font-size:0.9rem;color:#9aa0a6;text-align:center;margin-bottom:1rem;">✂️ Will transcribe: {format_time(start_time)} - {format_time(end_time)} ({format_time(trim_duration)} total)</div>', unsafe_allow_html=True)
        
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
            if model == "OpenAI" and not get_secret("OPENAI_API_KEY"):
                st.error("❌ OpenAI API key not found. Please enter it in a .env file.")
                return
            elif model == "Deepgram" and not get_secret("DEEPGRAM_API_KEY"):
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
                    # Get diarize setting (supported by both Deepgram and OpenAI)
                    model = st.session_state.get('model', 'Deepgram')
                    diarize_setting = st.session_state.get('diarize', False)
                    transcription = transcribe_large_file(
                        audio_path,
                        model,
                        language=language_code,
                        diarize=diarize_setting,
                        progress_bar=progress_bar,
                        status_text=status_text,
                    )
                    # Persist immediately, before any further Streamlit calls:
                    # if the browser session died mid-run, the next UI call may
                    # abort this script and the result would be lost.
                    _save_last_transcript(uploaded_file.name, transcription)
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
        # Full-width download button (better on mobile than a half-width column).
        # st.code() already provides a built-in copy button, so no second column is needed.
        download_data = st.session_state.transcription.encode('utf-8')
        st.download_button(
            label="Download",
            data=download_data,
            file_name=f"{st.session_state.filename.split('.')[0]}_transcript.txt",
            mime="text/plain",
            help="Download the transcription as a text file"
        )
    else:
        # No transcript in this session — offer the last one persisted
        # server-side (recovers results lost to a dropped mobile session).
        last = _load_last_transcript()
        if last:
            with st.expander(f"📄 Recuperar última transcripción — {last['filename']} ({last.get('saved_at', '')})"):
                st.code(last["transcription"], language=None)
                st.download_button(
                    label="Download",
                    data=last["transcription"].encode("utf-8"),
                    file_name=f"{last['filename'].split('.')[0]}_transcript.txt",
                    mime="text/plain",
                    key="download_recovered_transcript",
                    help="Download the recovered transcription as a text file"
                )

    # Model Selector (subtitle + select box, no bubble)
    # st.markdown('<div style="font-size:1.1rem;font-weight:600;margin-top:1.5rem;margin-bottom:0.5rem;">Model</div>', unsafe_allow_html=True)
    # Guard against a stale session value from before the "OpenAI Whisper" -> "OpenAI"
    # rename (Streamlit raises if the stored value isn't among the options)
    if st.session_state.get('model') not in ("Deepgram", "OpenAI"):
        st.session_state['model'] = "Deepgram"
    model = st.selectbox(
        "Transcription Model",
        ["Deepgram", "OpenAI"],
        key='model',
        help="Choose the transcription service to use. OpenAI uses gpt-4o-transcribe (with automatic fallback to lighter models)."
    )

    # Language Selector (Español/Inglés)
    language = st.selectbox(
        "Language",
        ["🇪🇸 Español", "🇬🇧 English"],
        key='language',
        help="Select the language of the audio for better transcription accuracy."
    )

    # Speaker Diarization Toggle
    diarize_enabled = st.checkbox(
        "Identify different speakers (diarization)",
        key='diarize',
        value=True,  # Default to enabled
        help="Enable speaker identification to show who said what. Works with both Deepgram and OpenAI."
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
