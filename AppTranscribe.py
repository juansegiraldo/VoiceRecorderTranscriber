import streamlit as st
import os
from openai import OpenAI
import requests
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import json
import shutil
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

# Shared transcription/diarization/analysis core (ROADMAP Phase 0 extraction —
# the same module backs scripts/deepgram_transcribe_cli.py). Imported after the
# FFmpeg patch on principle; core only lazy-imports pydub when splitting.
from core.transcription import (
    METRIC_GUIDE_ES,
    RATING_ICONS,
    WHISPER_MAX_CHUNK_MB,
    build_report,
    describe_sentiment_score,
    sentiment_timeline_emoji,
    split_audio_file,
    transcribe_file_deepgram,
)

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

def transcribe_with_openai(path: str, language: str = None) -> str:
    api_key = get_secret("OPENAI_API_KEY")
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


def transcribe_large_file_whisper(file_path: str, language: str = None, progress_bar=None, status_text=None) -> str:
    """OpenAI Whisper path: chunk to fit the 25 MB API limit and concatenate.

    The Deepgram path does NOT go through here — core.transcription handles
    size (much higher limit) and diarization/speaker consistency itself.
    """
    if status_text:
        status_text.text("🔍 Analyzing file...")
    chunk_paths = split_audio_file(file_path, max_size_mb=WHISPER_MAX_CHUNK_MB)
    if len(chunk_paths) == 1:
        if status_text:
            status_text.text("🎤 Transcribing file...")
        return transcribe_with_openai(file_path, language)
    if status_text:
        status_text.text(f"📦 File split into {len(chunk_paths)} chunks")
    transcriptions = []
    temp_dir = os.path.dirname(chunk_paths[0])
    try:
        for i, chunk_path in enumerate(chunk_paths, 1):
            if status_text:
                status_text.text(f"🎤 Processing chunk {i}/{len(chunk_paths)}...")
            if progress_bar:
                progress_bar.progress(i / len(chunk_paths))
            try:
                transcriptions.append(transcribe_with_openai(chunk_path, language))
                if status_text:
                    status_text.text(f"✅ Chunk {i} processed")
            except Exception as e:
                if status_text:
                    status_text.text(f"❌ Error processing chunk {i}: {str(e)}")
                transcriptions.append(f"[Error in chunk {i}: {str(e)}]")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    return " ".join(transcriptions)


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
            '<div style="text-align:center;color:#666;margin-bottom:0.8rem;">'
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
        local_file = st.file_uploader(
            "Audio file",
            type=allowed_types,
            help="Select an audio file to transcribe (MP3, WAV, M4A, MP4)" + (" (M4A/MP4 extraction requires FFmpeg)" if not ffmpeg_available else ""),
            label_visibility="collapsed"
        )
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
        st.markdown(f'<div style="font-size:0.9rem;color:#666;text-align:center;margin-bottom:1rem;">🎵 Duration: {format_time(duration_seconds)}</div>', unsafe_allow_html=True)
        
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
            language_ui = st.session_state.get('language', '🌐 Auto (detectar idioma)')
            # Mapear a código de idioma; None = el proveedor detecta el idioma
            # (Deepgram vía detect_language, Whisper de forma nativa).
            if 'auto' in language_ui.lower():
                language_code = None
            elif 'espa' in language_ui.lower():
                language_code = 'es'
            else:
                language_code = 'en'
            if model == "OpenAI Whisper" and not get_secret("OPENAI_API_KEY"):
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
                    model = st.session_state.get('model', 'Deepgram')
                    analysis = None
                    if model == "Deepgram":
                        # Deepgram path: core handles size/chunking, the model
                        # fallback chain, diarization v2 and optional analysis.
                        diarize_setting = st.session_state.get('diarize', False)
                        sentiment_setting = st.session_state.get('sentiment', False)
                        insights_setting = st.session_state.get('insights', True)

                        def _progress(fraction, message):
                            if fraction is not None and progress_bar:
                                progress_bar.progress(min(max(fraction, 0.0), 1.0))
                            if status_text:
                                status_text.text(message)

                        analysis = transcribe_file_deepgram(
                            audio_path,
                            get_secret("DEEPGRAM_API_KEY"),
                            language=language_code,
                            diarize=diarize_setting,
                            want_sentiment=sentiment_setting,
                            want_insights=insights_setting,
                            progress_cb=_progress,
                        )
                        transcription = analysis["transcript"]
                    else:
                        transcription = transcribe_large_file_whisper(
                            audio_path,
                            language=language_code,
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
                st.session_state.analysis = analysis
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

        # Conversation analysis (Deepgram only). Rendered as text-labeled stat
        # rows + native meters — identity always carried by text, never color.
        analysis = st.session_state.get('analysis')
        if analysis:
            meta_bits = []
            if analysis.get("detected_language"):
                meta_bits.append(f"🌐 Idioma detectado: {analysis['detected_language']}")
            if analysis.get("model_used"):
                meta_bits.append(f"modelo: {analysis['model_used']}")
            if meta_bits:
                st.caption(" · ".join(meta_bits))
            for warning in analysis.get("warnings", []):
                st.caption(f"⚠️ {warning}")

            insights = analysis.get("insights")
            if insights:
                with st.expander("📊 Métricas de conversación", expanded=True):
                    for item in insights["feedback"]:
                        st.markdown(f"- {item}")
                    st.markdown("---")
                    for speaker, s in sorted(insights["per_speaker"].items(), key=lambda kv: str(kv[0])):
                        wpm_txt = f" · {s['wpm']:.0f} palabras/min" if s.get('wpm') else ""
                        st.markdown(
                            f"**Speaker {speaker}** — {s['talk_share'] * 100:.0f}% del habla · "
                            f"{format_time(s['talk_time'])} · {s['words']} palabras{wpm_txt}"
                        )
                        st.progress(min(max(s['talk_share'], 0.0), 1.0))
                        rating_notes = " · ".join(
                            f"{RATING_ICONS.get(r['level'], 'ℹ️')} {r['note']}"
                            for r in (s.get("pace_rating"), s.get("fillers_rating")) if r
                        )
                        if rating_notes:
                            st.caption(rating_notes)
                    overall = insights["overall"]
                    st.caption(
                        f"{overall['n_speakers']} hablante(s) · {overall['total_words']} palabras · "
                        f"{overall['interruptions']} interrupciones · "
                        f"{overall['silence_ratio'] * 100:.0f}% de silencio · "
                        f"{overall['questions']} preguntas"
                    )

            sentiment = analysis.get("sentiment")
            if sentiment:
                with st.expander("😊 Sentimiento (Deepgram)", expanded=True):
                    avg = sentiment["average"]
                    label_es = {"positive": "😊 Positivo", "neutral": "😐 Neutral",
                                "negative": "🙁 Negativo"}.get(avg["sentiment"], avg["sentiment"])
                    st.markdown(
                        f"**Tono general:** {label_es} ({avg['sentiment_score']:+.2f}) — "
                        f"{describe_sentiment_score(avg['sentiment_score'])}."
                    )
                    st.caption(
                        "Escala −1…+1 · Deepgram etiqueta positivo a partir de +0.33 "
                        "y negativo por debajo de −0.33."
                    )
                    emoji_line = sentiment_timeline_emoji(sentiment)
                    if emoji_line:
                        st.markdown(f"**Evolución** (inicio → fin): {emoji_line}")
                    for speaker, sp in sorted(sentiment["per_speaker"].items(), key=lambda kv: str(kv[0])):
                        lab = {"positive": "positivo", "neutral": "neutral",
                               "negative": "negativo"}.get(sp["sentiment"], sp["sentiment"])
                        st.markdown(
                            f"- Speaker {speaker}: {lab} ({sp['sentiment_score']:+.2f} — "
                            f"{describe_sentiment_score(sp['sentiment_score'])})"
                        )

            if insights or sentiment:
                with st.expander("ℹ️ Cómo leer estas métricas"):
                    for guide_line in METRIC_GUIDE_ES:
                        st.markdown(f"- {guide_line}")

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
        if analysis and (analysis.get("insights") or analysis.get("sentiment")):
            report_data = build_report(
                analysis, filename=st.session_state.get('filename', '')
            ).encode('utf-8')
            st.download_button(
                label="📄 Descargar informe (transcript + análisis)",
                data=report_data,
                file_name=f"{st.session_state.filename.split('.')[0]}_report.txt",
                mime="text/plain",
                help="Transcripción más métricas de conversación y sentimiento en un solo archivo"
            )
    # If no transcription, do not show the frame, placeholder, or empty text area

    # Model Selector (subtitle + select box, no bubble)
    # st.markdown('<div style="font-size:1.1rem;font-weight:600;margin-top:1.5rem;margin-bottom:0.5rem;">Model</div>', unsafe_allow_html=True)
    model = st.selectbox(
        "Transcription Model",
        ["Deepgram", "OpenAI Whisper"],
        key='model',
        help="Choose the transcription service to use. OpenAI Whisper often works better for non-English content."
    )

    # Language Selector (Auto/Español/Inglés)
    language = st.selectbox(
        "Language",
        ["🌐 Auto (detectar idioma)", "🇪🇸 Español", "🇬🇧 English"],
        key='language',
        help="Con 'Auto' el proveedor detecta el idioma solo (elige UN idioma dominante). "
             "Fijar el idioma explícito afina algo la precisión en audios muy mezclados. "
             "El análisis de sentimiento solo funciona con audio en inglés."
    )

    # Speaker Diarization Toggle (only for Deepgram)
    diarize_enabled = st.checkbox(
        "Identify different speakers (diarization)",
        key='diarize',
        value=True,  # Default to enabled
        disabled=(model == "OpenAI Whisper"),
        help="Enable speaker identification to show who said what. Only works with Deepgram model."
    )

    # Conversation analysis toggles (Deepgram only)
    insights_enabled = st.checkbox(
        "📊 Métricas de conversación",
        key='insights',
        value=True,
        disabled=(model == "OpenAI Whisper"),
        help="Tiempo de habla por hablante, ritmo (palabras/min), interrupciones, "
             "monólogos y muletillas, con observaciones. Funciona en español e inglés."
    )
    sentiment_enabled = st.checkbox(
        "😊 Análisis de sentimiento (solo audio en inglés)",
        key='sentiment',
        value=False,
        disabled=(model == "OpenAI Whisper"),
        help="Análisis de sentimiento de Deepgram (positivo/neutral/negativo por tramo "
             "y por hablante). Deepgram solo lo ofrece para audio en inglés; "
             "para español se omite con un aviso."
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
