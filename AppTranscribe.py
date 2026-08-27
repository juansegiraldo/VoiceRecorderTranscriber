import streamlit as st
import os
from openai import OpenAI
import requests
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import hmac
import html
import json
import re
import shutil
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
    WHISPER_MAX_CHUNK_MB,
    build_report,
    describe_sentiment_score,
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

# Page configuration. page_icon is the browser favicon (not in-app UI) — the
# only emoji that survives the v2 redesign besides the plain-text report.
st.set_page_config(
    page_title="VoiceTranscriber",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
        status_text.text("Analizando el archivo…")
    chunk_paths = split_audio_file(file_path, max_size_mb=WHISPER_MAX_CHUNK_MB)
    if len(chunk_paths) == 1:
        if status_text:
            status_text.text("Transcribiendo con Whisper…")
        return transcribe_with_openai(file_path, language)
    if status_text:
        status_text.text(f"Archivo dividido en {len(chunk_paths)} partes")
    transcriptions = []
    temp_dir = os.path.dirname(chunk_paths[0])
    try:
        for i, chunk_path in enumerate(chunk_paths, 1):
            if status_text:
                status_text.text(f"Procesando parte {i} de {len(chunk_paths)}…")
            if progress_bar:
                progress_bar.progress(i / len(chunk_paths))
            try:
                transcriptions.append(transcribe_with_openai(chunk_path, language))
                if status_text:
                    status_text.text(f"Parte {i} lista")
            except Exception as e:
                if status_text:
                    status_text.text(f"Error en la parte {i}: {str(e)}")
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
        st.error("El inicio de sesión con Google expiró o no se pudo validar. Pulsa el botón de nuevo.")
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
        st.error(f"No se pudo completar el inicio de sesión con Google: {e}")
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
            "Google Drive no está configurado en este despliegue. "
            "Añade `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` y `GOOGLE_REDIRECT_URI` "
            "en los *secrets* para habilitarlo."
        )
        return None

    creds = _get_drive_credentials()
    if creds is None:
        st.caption("Inicia sesión con Google para elegir un audio de tu Drive.")
        try:
            login_url = _get_drive_login_url()
            st.link_button("Iniciar sesión con Google", login_url, use_container_width=True)
        except Exception as e:
            st.error(f"No se pudo iniciar el login de Google: {e}")
        return None

    # Authenticated: list audio and let the user pick + load one.
    try:
        files = list_drive_audio(creds)
    except Exception as e:
        logger.error(f"Drive list failed: {e}")
        st.error(f"No se pudieron listar los archivos de Drive: {e}")
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
        if st.button("Cargar de Drive", type="primary", key="drive_load_btn", use_container_width=True):
            with st.spinner("Descargando de Google Drive…"):
                try:
                    loaded = download_drive_file(
                        creds, selected["id"], selected["name"], selected.get("mimeType")
                    )
                    st.session_state.drive_loaded_id = selected["id"]
                except Exception as e:
                    logger.error(f"Drive download failed: {e}")
                    st.error(f"No se pudo descargar el archivo: {e}")
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


# ---------------------------------------------------------------------------
# v2 presentation layer (see docs/rediseno-ux-v2.md and the mockup in
# docs/mockups/rediseno-v2.html). UI-only: core/transcription.py is untouched;
# these helpers reshape its output for the screen. The downloadable .txt report
# (build_report) keeps its emoji-based ratings — in plain text there is no
# color, so there the emoji still does real work.
# ---------------------------------------------------------------------------

# Speaker colors: fixed order, CVD-validated on white. Identity is always
# carried by the text label next to the swatch, never by color alone.
SPEAKER_COLORS = ["#2A78D6", "#008300", "#E87BA4", "#EDA100"]

# Visible label -> provider language code (None = auto-detect). Replaces the
# old fragile substring sniffing ('auto' in label.lower()).
LANGUAGE_OPTIONS = {
    "Automático (detectar idioma)": None,
    "Español": "es",
    "Inglés": "en",
}

# The option VALUES "Deepgram" / "OpenAI Whisper" are load-bearing (the
# pipeline compares those exact strings); only the visible label changes.
MODEL_LABELS = {"Deepgram": "Deepgram — recomendado", "OpenAI Whisper": "OpenAI Whisper"}

LANG_NAMES = {"es": "Español", "en": "Inglés"}
RATING_WORDS = {"good": "Bien", "ok": "Justo", "warn": "Mejorable"}
RATING_CSS = {"good": "b-good", "ok": "b-ok", "warn": "b-warn"}
SENTIMENT_WORDS = {"positive": "Positivo", "neutral": "Neutral", "negative": "Negativo"}
SENTIMENT_TEXT_COLORS = {"positive": "#1C5CAB", "neutral": "#4D5A55", "negative": "#B23636"}

# Characters stripped from the FRONT of core-generated strings before showing
# them in the UI (the .txt report keeps them).
_UI_EMOJI_CHARS = "✅🟡⚠ℹ🔍🎤🎙📦😊😐🙁️ "

_SPEAKER_RE = re.compile(r"\bSpeaker (\d+)\b")

_PROGRESS_ES = [
    (re.compile(r"Analyzing file"), "Analizando el archivo…"),
    (re.compile(r"Transcribing with Deepgram"), "Transcribiendo con Deepgram…"),
    (re.compile(r"File split into (\d+) chunks"), r"Archivo dividido en \1 partes"),
    (re.compile(r"Processing chunk (\d+)/(\d+)"), r"Procesando parte \1 de \2…"),
]


def strip_ui_emoji(text) -> str:
    return str(text).lstrip(_UI_EMOJI_CHARS)


def humanize_feedback(text) -> str:
    """Core feedback bullets say '⚠️ Speaker 0 …'; on screen we want
    'Hablante 1 …' (1-based, Spanish, no emoji — the evaluation lives in the
    words themselves)."""
    return _SPEAKER_RE.sub(
        lambda m: f"Hablante {int(m.group(1)) + 1}", strip_ui_emoji(text)
    )


def progress_message_es(message) -> str:
    """Translate the few known core progress messages; fall back to the
    emoji-stripped original for anything new."""
    message = strip_ui_emoji(message)
    for pat, repl in _PROGRESS_ES:
        m = pat.search(message)
        if m:
            return m.expand(repl)
    return message


def speaker_name(speaker) -> str:
    """Deepgram ids are 0-based ints; people count from 1."""
    try:
        return f"Hablante {int(speaker) + 1}"
    except (TypeError, ValueError):
        return f"Hablante {speaker}"


def speaker_color(speaker, ordered_speakers) -> str:
    try:
        idx = list(ordered_speakers).index(speaker)
    except ValueError:
        idx = 0
    return SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]


def fmt_int_es(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def fmt_dec_es(x) -> str:
    return f"{x:.1f}".replace(".", ",")


def fmt_signed_es(x) -> str:
    return f"{x:+.2f}".replace(".", ",")


def rating_badge(rating) -> str:
    """dict {'level', 'note'} -> small text chip (or '')."""
    if not rating:
        return ""
    word = RATING_WORDS.get(rating.get("level"))
    if not word:
        return ""
    css = RATING_CSS.get(rating.get("level"), "b-ok")
    return f'<span class="vt-badge {css}">{word}</span>'


def group_turns(utterances) -> list[dict]:
    """Collapse consecutive same-speaker utterances into display turns."""
    turns = []
    for u in utterances or []:
        text = (u.get("transcript") or "").strip()
        if not text:
            continue
        sp = u.get("speaker", 0)
        if turns and turns[-1]["speaker"] == sp:
            turns[-1]["text"] += " " + text
        else:
            turns.append({"speaker": sp, "start": u.get("start") or 0.0, "text": text})
    return turns


def _sentiment_seg_color(score):
    """Diverging scale for the timeline: blue positive <-> red negative,
    neutral gray in the middle, None -> empty segment."""
    if score is None:
        return None
    if score >= 0.5:
        return "#1C5CAB"
    if score >= 0.33:
        return "#2A78D6"
    if score >= 0.1:
        return "#86B6EF"
    if score > -0.1:
        return "#ECEEED"
    if score > -0.33:
        return "#F0A1A0"
    if score > -0.5:
        return "#E34948"
    return "#B23636"


V2_CSS = """
<style>
html { -webkit-text-size-adjust: 100%; }
/* 4.4rem clears Streamlit's fixed header without the default 6rem chasm */
.block-container { max-width: 720px !important; margin: 0 auto; padding-top: 4.4rem !important; }
@media (max-width: 600px) {
  .block-container { padding-left: 0.9rem !important; padding-right: 0.9rem !important; }
}
/* Mobile-first tap targets (kept from v1) */
.stButton > button, .stDownloadButton > button { width: 100%; min-height: 44px; }
div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
  min-height: 44px; font-size: 16px !important; /* >=16px stops iOS zoom-on-focus */
}
.stFileUploader { width: 100%; }
/* Brand */
.vt-brand { display: flex; align-items: center; justify-content: center; gap: 9px;
  font-size: 1.45rem; font-weight: 700; letter-spacing: -0.01em; margin: 0.1rem 0 0.15rem; }
.vt-wave { display: inline-flex; align-items: flex-end; gap: 2.5px; height: 17px; }
.vt-wave i { width: 3.5px; border-radius: 2px; background: #0E7263; display: block; }
.vt-wave i:nth-child(1) { height: 55%; }
.vt-wave i:nth-child(2) { height: 100%; }
.vt-wave i:nth-child(3) { height: 35%; }
.vt-tagline { text-align: center; color: #6A7772; font-size: 0.9rem; margin-bottom: 1rem; }
/* File card */
.vt-file { background: #fff; border: 1px solid #E4E8E6; border-radius: 12px; padding: 0.7rem 0.9rem; }
.vt-file-name { font-weight: 600; word-break: break-all; }
.vt-file-meta { color: #6A7772; font-size: 0.86rem; margin-top: 1px; }
.vt-summary { color: #6A7772; font-size: 0.86rem; line-height: 1.35; }
/* Result header */
.vt-h1 { font-weight: 700; font-size: 1.15rem; margin: 0.4rem 0 0.1rem; word-break: break-all; }
.vt-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.35rem 0 0.5rem; }
.vt-chip { background: #F1F4F2; color: #4D5A55; border-radius: 999px; padding: 2px 10px;
  font-size: 0.8rem; font-weight: 600; }
/* Transcript turns */
.vt-turn { padding: 0.55rem 0; border-bottom: 1px solid #EEF1EF; }
.vt-turn:last-child { border-bottom: none; }
.vt-who { display: flex; align-items: center; gap: 7px; font-weight: 600; font-size: 0.9rem; }
.vt-dot { width: 10px; height: 10px; border-radius: 3px; flex: none; }
.vt-ts { font-family: monospace; font-size: 0.75rem; color: #9AA7A1; font-weight: 400; }
.vt-turn p { margin: 0.25rem 0 0; line-height: 1.55; }
.vt-plain { white-space: pre-wrap; line-height: 1.55; background: #fff;
  border: 1px solid #E4E8E6; border-radius: 12px; padding: 0.8rem 1rem; }
/* Metric tiles */
.vt-tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 0.4rem 0 0.2rem; }
@media (max-width: 480px) { .vt-tiles { grid-template-columns: repeat(2, 1fr); } }
.vt-tile { background: #fff; border: 1px solid #E4E8E6; border-radius: 10px; padding: 0.55rem 0.7rem; }
.vt-tile .v { font-size: 1.25rem; font-weight: 700; font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em; }
.vt-tile .l { font-size: 0.74rem; color: #6A7772; margin-top: 1px; }
/* Text badges (replace the on-screen emoji ratings) */
.vt-badge { display: inline-block; border-radius: 6px; padding: 1px 7px; font-size: 0.72rem;
  font-weight: 700; vertical-align: 1px; }
.b-good { background: #E5F3E5; color: #175617; }
.b-ok { background: #FBF0D7; color: #7A4E00; }
.b-warn { background: #FBE7DE; color: #8C3E1A; }
/* Talk-share bars */
.vt-sect { font-weight: 600; font-size: 0.95rem; margin: 1rem 0 0.4rem; }
.vt-bar { display: grid; grid-template-columns: 108px 1fr 44px; align-items: center; gap: 9px; margin: 5px 0; }
.vt-bar .n { display: flex; align-items: center; gap: 6px; font-size: 0.84rem; font-weight: 600; }
.vt-bar .t { background: #F1F4F2; border-radius: 5px; height: 13px; overflow: hidden; }
.vt-bar .t i { display: block; height: 100%; border-radius: 0 4px 4px 0; }
.vt-bar .v { font-size: 0.84rem; font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }
/* Per-speaker cards */
.vt-spcard { background: #fff; border: 1px solid #E4E8E6; border-radius: 10px;
  padding: 0.6rem 0.8rem; margin-top: 7px; font-size: 0.86rem; color: #4D5A55; }
.vt-spcard .vt-who { margin-bottom: 2px; color: inherit; }
.vt-spnote { font-size: 0.78rem; color: #6A7772; margin-top: 3px; }
/* Sentiment */
.vt-tone { display: flex; align-items: baseline; gap: 9px; margin-top: 0.4rem; }
.vt-tone .w { font-size: 1.4rem; font-weight: 700; }
.vt-tone .s { color: #6A7772; font-variant-numeric: tabular-nums; }
.vt-tl { display: flex; gap: 2px; height: 26px; margin: 6px 0 3px; }
.vt-tl i { flex: 1; border-radius: 3px; }
.vt-tl i.empty { background: transparent; border: 1px dashed #D5DBD8; }
.vt-tlx { display: flex; justify-content: space-between; font-family: monospace;
  font-size: 0.72rem; color: #9AA7A1; }
.vt-leg { display: flex; gap: 14px; font-size: 0.8rem; color: #4D5A55; margin-top: 6px; flex-wrap: wrap; }
.vt-leg span { display: inline-flex; align-items: center; gap: 5px; }
.vt-leg i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
/* Footer */
.vt-footer { text-align: center; color: #9AA7A1; font-size: 0.82rem; line-height: 1.7; margin-top: 0.6rem; }
.vt-footer a { color: #0E7263; }
</style>
"""


def render_settings_row():
    """One-line summary + an 'Ajustes' popover. Rendered BEFORE the CTA so the
    widget keys ('model', 'language', 'diarize', 'insights', 'sentiment') are
    populated when the transcription handler reads them from session_state."""
    model_now = st.session_state.get("model", "Deepgram")
    lang_label = st.session_state.get("language", next(iter(LANGUAGE_OPTIONS)))
    lang_short = lang_label.split(" (")[0].lower()
    diar_txt = "diarización sí" if st.session_state.get("diarize", True) else "diarización no"
    summary = f"{model_now} · idioma {lang_short} · {diar_txt}"

    col_sum, col_pop = st.columns([2.4, 1], vertical_alignment="center")
    with col_sum:
        st.markdown(f'<div class="vt-summary">{html.escape(summary)}</div>', unsafe_allow_html=True)
    with col_pop:
        with st.popover("Ajustes", use_container_width=True):
            model = st.selectbox(
                "Modelo de transcripción",
                list(MODEL_LABELS),
                key="model",
                format_func=lambda v: MODEL_LABELS.get(v, v),
                help="Deepgram añade hablantes, métricas y sentimiento; Whisper devuelve solo texto.",
            )
            st.selectbox(
                "Idioma del audio",
                list(LANGUAGE_OPTIONS),
                key="language",
                help="Con «Automático» el proveedor detecta el idioma dominante. "
                     "Fijarlo afina algo la precisión en audios muy mezclados.",
            )
            whisper = model == "OpenAI Whisper"
            st.toggle(
                "Identificar hablantes (diarización)",
                key="diarize", value=True, disabled=whisper,
                help="Muestra quién dijo qué. Solo con Deepgram.",
            )
            st.toggle(
                "Métricas de conversación",
                key="insights", value=True, disabled=whisper,
                help="Reparto del habla, ritmo, interrupciones, monólogos y muletillas, "
                     "con observaciones. Funciona en español e inglés.",
            )
            st.toggle(
                "Análisis de sentimiento",
                key="sentiment", value=False, disabled=whisper,
                help="Deepgram solo lo ofrece para audio en inglés; en español se omite con un aviso.",
            )


def render_transcript_tab(transcription: str, analysis: dict | None):
    turns = group_turns((analysis or {}).get("utterances"))
    if turns:
        order = sorted({t["speaker"] for t in turns}, key=str)
        parts = []
        for t in turns:
            color = speaker_color(t["speaker"], order)
            parts.append(
                f'<div class="vt-turn"><div class="vt-who">'
                f'<span class="vt-dot" style="background:{color}"></span>'
                f'{html.escape(speaker_name(t["speaker"]))}'
                f'<span class="vt-ts">{format_time(t["start"])}</span></div>'
                f'<p>{html.escape(t["text"])}</p></div>'
            )
        st.markdown(f'<div class="vt-transcript">{"".join(parts)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="vt-plain">{html.escape(transcription)}</div>', unsafe_allow_html=True)


def render_metrics_tab(insights: dict):
    o = insights["overall"]
    tiles = [
        (str(o["n_speakers"]), "hablantes", None),
        (fmt_int_es(o["total_words"]), "palabras", None),
        (f"{o['wpm']:.0f}" if o.get("wpm") else "—", "palabras/min", None),
        (str(o["interruptions"]), "interrupciones", None),
        (f"{o['silence_ratio'] * 100:.0f}%", "de silencio", o.get("silence_rating")),
        (str(o["questions"]), "preguntas", None),
    ]
    tile_html = "".join(
        f'<div class="vt-tile"><div class="v">{v}</div><div class="l">{l} {rating_badge(r)}</div></div>'
        for v, l, r in tiles
    )
    st.markdown(f'<div class="vt-tiles">{tile_html}</div>', unsafe_allow_html=True)

    ordered = sorted(insights["per_speaker"].items(), key=lambda kv: str(kv[0]))
    order_ids = [sp for sp, _ in ordered]

    bars = []
    for sp, s in ordered:
        color = speaker_color(sp, order_ids)
        share = min(max(s.get("talk_share", 0.0), 0.0), 1.0)
        bars.append(
            f'<div class="vt-bar"><span class="n">'
            f'<span class="vt-dot" style="background:{color}"></span>{html.escape(speaker_name(sp))}</span>'
            f'<div class="t"><i style="width:{share * 100:.0f}%;background:{color}"></i></div>'
            f'<span class="v">{share * 100:.0f}%</span></div>'
        )
    st.markdown('<div class="vt-sect">Reparto del habla</div>' + "".join(bars), unsafe_allow_html=True)

    cards = []
    for sp, s in ordered:
        color = speaker_color(sp, order_ids)
        stats = [f"{format_time(s['talk_time'])} en uso de palabra", f"{fmt_int_es(s['words'])} palabras"]
        if s.get("wpm"):
            stats.append(f"{s['wpm']:.0f} ppm {rating_badge(s.get('pace_rating'))}")
        stats.append(
            f"muletillas {fmt_dec_es(s.get('fillers_per_100_words', 0.0))}/100 "
            f"{rating_badge(s.get('fillers_rating'))}"
        )
        notes = " · ".join(
            html.escape(r["note"]) for r in (s.get("pace_rating"), s.get("fillers_rating")) if r
        )
        cards.append(
            f'<div class="vt-spcard"><div class="vt-who">'
            f'<span class="vt-dot" style="background:{color}"></span>{html.escape(speaker_name(sp))}</div>'
            f'{" · ".join(stats)}'
            + (f'<div class="vt-spnote">{notes}</div>' if notes else "")
            + "</div>"
        )
    st.markdown("".join(cards), unsafe_allow_html=True)

    if insights.get("feedback"):
        st.markdown('<div class="vt-sect">Observaciones</div>', unsafe_allow_html=True)
        for item in insights["feedback"]:
            st.markdown(f"- {humanize_feedback(item)}")

    with st.expander("Cómo leer estas métricas"):
        for guide_line in METRIC_GUIDE_ES:
            st.markdown(f"- {guide_line}")


def render_sentiment_tab(sentiment: dict):
    avg = sentiment["average"]
    score = avg["sentiment_score"]
    word = SENTIMENT_WORDS.get(avg["sentiment"], str(avg["sentiment"]))
    color = SENTIMENT_TEXT_COLORS.get(avg["sentiment"], "#4D5A55")
    st.markdown(
        f'<div class="vt-tone"><span class="w" style="color:{color}">{html.escape(word)}</span>'
        f'<span class="s">{fmt_signed_es(score)}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"Tono general de la conversación: {describe_sentiment_score(score)}.")

    timeline = sentiment.get("timeline") or []
    if timeline:
        segs = []
        for b in timeline:
            seg_color = _sentiment_seg_color(b.get("score"))
            label = f'{format_time(b.get("start", 0))}–{format_time(b.get("end", 0))}'
            if b.get("score") is not None:
                label += f': {fmt_signed_es(b["score"])}'
            else:
                label += ": sin datos"
            if seg_color is None:
                segs.append(f'<i class="empty" title="{label}"></i>')
            else:
                segs.append(f'<i style="background:{seg_color}" title="{label}"></i>')
        st.markdown(
            '<div class="vt-sect">Evolución (inicio → fin)</div>'
            f'<div class="vt-tl">{"".join(segs)}</div>'
            f'<div class="vt-tlx"><span>{format_time(timeline[0].get("start", 0))}</span>'
            f'<span>{format_time(timeline[-1].get("end", 0))}</span></div>'
            '<div class="vt-leg">'
            '<span><i style="background:#2A78D6"></i>Positivo</span>'
            '<span><i style="background:#ECEEED;border:1px solid #D5DBD8"></i>Neutral</span>'
            '<span><i style="background:#E34948"></i>Negativo</span></div>',
            unsafe_allow_html=True,
        )

    per_speaker = sentiment.get("per_speaker") or {}
    if per_speaker:
        st.markdown('<div class="vt-sect">Por hablante</div>', unsafe_allow_html=True)
        rows = []
        for sp, spdata in sorted(per_speaker.items(), key=lambda kv: str(kv[0])):
            sp_word = SENTIMENT_WORDS.get(spdata["sentiment"], str(spdata["sentiment"]))
            rows.append(
                f'<div class="vt-spcard"><div class="vt-who">{html.escape(speaker_name(sp))}</div>'
                f'{html.escape(sp_word)} · {fmt_signed_es(spdata["sentiment_score"])} — '
                f'{html.escape(describe_sentiment_score(spdata["sentiment_score"]))}</div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

    st.caption(
        "Escala −1…+1 · positivo desde +0,33 y negativo desde −0,33. "
        "Deepgram solo analiza sentimiento en audio en inglés."
    )


def render_results():
    transcription = st.session_state.transcription
    analysis = st.session_state.get("analysis")
    filename = st.session_state.get("filename", "audio")
    base = filename.split(".")[0]
    insights = (analysis or {}).get("insights")
    sentiment = (analysis or {}).get("sentiment")

    st.markdown(f'<div class="vt-h1">{html.escape(base)}</div>', unsafe_allow_html=True)
    chips = []
    if analysis:
        detected = analysis.get("detected_language")
        if detected:
            chips.append(f"{LANG_NAMES.get(detected, detected)} · detectado")
        if analysis.get("model_used"):
            chips.append(str(analysis["model_used"]))
        duration = analysis.get("duration") or (insights or {}).get("overall", {}).get("duration")
        if duration:
            chips.append(format_time(duration))
    if chips:
        st.markdown(
            '<div class="vt-chips">'
            + "".join(f'<span class="vt-chip">{html.escape(c)}</span>' for c in chips)
            + "</div>",
            unsafe_allow_html=True,
        )
    for warning in (analysis or {}).get("warnings", []):
        st.caption(f"Aviso: {strip_ui_emoji(warning)}")

    tab_names = (
        ["Transcripción"]
        + (["Métricas"] if insights else [])
        + (["Sentimiento"] if sentiment else [])
    )
    if len(tab_names) == 1:
        render_transcript_tab(transcription, analysis)
    else:
        tabs = st.tabs(tab_names)
        with tabs[0]:
            render_transcript_tab(transcription, analysis)
        next_tab = 1
        if insights:
            with tabs[next_tab]:
                render_metrics_tab(insights)
            next_tab += 1
        if sentiment:
            with tabs[next_tab]:
                render_sentiment_tab(sentiment)

    download_data = transcription.encode("utf-8")
    if analysis and (insights or sentiment):
        report_data = build_report(analysis, filename=filename).encode("utf-8")
        col_txt, col_rep = st.columns(2)
        with col_txt:
            st.download_button(
                "Descargar .txt", data=download_data,
                file_name=f"{base}_transcript.txt", mime="text/plain",
                use_container_width=True,
            )
        with col_rep:
            st.download_button(
                "Descargar informe completo", data=report_data,
                file_name=f"{base}_report.txt", mime="text/plain",
                use_container_width=True,
                help="Transcripción + métricas y sentimiento en un solo archivo de texto",
            )
    else:
        st.download_button(
            "Descargar .txt", data=download_data,
            file_name=f"{base}_transcript.txt", mime="text/plain",
            use_container_width=True,
        )
    st.button(
        "Generar acta", disabled=True, use_container_width=True,
        help="Llega con la Fase 1 del roadmap: acta estructurada con IA "
             "a partir de la transcripción y tus notas.",
    )
    st.caption("«Generar acta» llega con la Fase 1 del roadmap; de momento puedes usar el GPT del pie de página.")


def _password_gate() -> bool:
    """Block the whole app behind a single shared password.

    The app is deployed at a public URL but is only meant for two people, so
    this is a private-by-default door, not a user system: one secret
    (APP_PASSWORD) shared by both. Returns True when the visitor is allowed in.

    The unlocked flag lives in st.session_state, so it lasts exactly as long as
    the browser tab: a refresh or a later visit asks again. If APP_PASSWORD is
    not configured the app stays OPEN (fail-open) — that keeps local dev and
    the CLI scripts working without a secret, and the Streamlit Cloud secret is
    what actually closes the public deployment.
    """
    if not get_secret("APP_PASSWORD"):
        return True  # no password configured (local dev) -> no gate
    if st.session_state.get("auth_ok"):
        return True

    st.markdown(
        '<div class="vt-brand"><span class="vt-wave"><i></i><i></i><i></i></span>'
        'VoiceTranscriber</div>'
        '<div class="vt-tagline">Acceso privado</div>',
        unsafe_allow_html=True,
    )

    # A form so that Enter submits (mobile keyboards show "go" instead of a
    # newline) and the password isn't re-checked on every keystroke rerun.
    with st.form("login", clear_on_submit=True):
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Entrar", type="primary")

    if submitted:
        # compare_digest keeps the check constant-time; str() guards against a
        # non-string secret (TOML would hand us an int for an all-digit value).
        if hmac.compare_digest(password, str(get_secret("APP_PASSWORD"))):
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")

    return False


def main():
    st.markdown(V2_CSS, unsafe_allow_html=True)

    # Private deployment: nothing below renders until the shared password is in.
    if not _password_gate():
        return

    # Complete any pending Google Drive OAuth redirect before rendering the UI.
    _handle_drive_oauth_callback()

    st.markdown(
        '<div class="vt-brand"><span class="vt-wave"><i></i><i></i><i></i></span>'
        'VoiceTranscriber</div>'
        '<div class="vt-tagline">Transcribe y analiza tus reuniones</div>',
        unsafe_allow_html=True,
    )

    allowed_types = ['mp3', 'wav']
    if ffmpeg_available:
        allowed_types.append('m4a')
        allowed_types.append('mp4')

    # Two input sources: a local upload or a file from the user's Google Drive.
    # Both converge on `uploaded_file` (a real UploadedFile or a DriveFile);
    # everything downstream is source-agnostic.
    tab_local, tab_drive = st.tabs(["Archivo", "Google Drive"])
    with tab_local:
        # The uploader key embeds a nonce so "Quitar" can clear the selection
        # by remounting the widget (Streamlit has no programmatic clear).
        local_file = st.file_uploader(
            "Archivo de audio",
            type=allowed_types,
            help="MP3, WAV, M4A o MP4"
                 + ("" if ffmpeg_available else " — M4A/MP4 no disponibles en este servidor (falta FFmpeg)"),
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.get('uploader_nonce', 0)}",
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
                    with st.spinner("Convirtiendo M4A a MP3…"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_tmp_file:
                            mp3_file_path = mp3_tmp_file.name
                        converted_path = convert_m4a_to_mp3(tmp_file_path, mp3_tmp_file.name)
                        os.unlink(tmp_file_path)
                        audio_path = converted_path
                        st.session_state.converted_mp3_path = audio_path
                        st.session_state.converted_mp3_file_key = file_key
                elif file_extension == 'mp4':
                    with st.spinner("Extrayendo el audio del MP4…"):
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
                st.error(f"No se pudo leer el archivo de audio: {str(e)}")
                trim_settings = None
                return
        audio_info = st.session_state.audio_info
        duration_seconds = audio_info['duration_seconds']

        # File card with a remove action ("Quitar" clears the selection by
        # remounting the uploader through its nonce key).
        size_mb = uploaded_file.size / 1024 / 1024
        card_col, remove_col = st.columns([4, 1], vertical_alignment="center")
        with card_col:
            st.markdown(
                f'<div class="vt-file"><div class="vt-file-name">{html.escape(uploaded_file.name)}</div>'
                f'<div class="vt-file-meta">{format_time(duration_seconds)} · {fmt_dec_es(size_mb)} MB</div></div>',
                unsafe_allow_html=True,
            )
        with remove_col:
            if st.button("Quitar", key="remove_file", use_container_width=True):
                st.session_state.uploader_nonce = st.session_state.get("uploader_nonce", 0) + 1
                for stale_key in ("drive_loaded_id", "drive_download_cache",
                                  "last_input_source", "current_file_key", "audio_info"):
                    st.session_state.pop(stale_key, None)
                st.rerun()

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

        # Trimming lives collapsed; the expander label carries the current
        # state (on_change callbacks ran before this line, so it's fresh).
        s_now, e_now = st.session_state.trim_range_slider
        is_trimmed = s_now > 0.05 or e_now < max_seconds - 0.05
        trim_label = (
            f"Recortar audio — {format_time(s_now)} a {format_time(e_now)}"
            + (" (recortado)" if is_trimmed else " (completo)")
        )
        with st.expander(trim_label, expanded=False):
            # Coarse selection: two-handle range slider
            st.slider(
                "Rango a transcribir",
                min_value=0.0,
                max_value=max_seconds,
                step=0.1,
                format="%.1f s",
                key="trim_range_slider",
                on_change=_sync_from_slider,
            )
            # Precise entry (much easier than dragging on a phone)
            num_col1, num_col2 = st.columns(2)
            with num_col1:
                st.number_input(
                    "Inicio (s)",
                    min_value=0.0,
                    max_value=max_seconds,
                    step=0.1,
                    format="%.1f",
                    key="trim_start_input",
                    on_change=_sync_from_numbers,
                )
            with num_col2:
                st.number_input(
                    "Fin (s)",
                    min_value=0.0,
                    max_value=max_seconds,
                    step=0.1,
                    format="%.1f",
                    key="trim_end_input",
                    on_change=_sync_from_numbers,
                )
            s_prev, e_prev = st.session_state.trim_range_slider
            st.caption(
                f"Se transcribirá {format_time(s_prev)} – {format_time(e_prev)} "
                f"({format_time(e_prev - s_prev)} en total)"
            )

        start_time, end_time = st.session_state.trim_range_slider

        trim_settings = {
            'start_time_ms': int(start_time * 1000),
            'end_time_ms': int(end_time * 1000),
            'duration_ms': audio_info['duration_ms']
        }
    
    # Settings render BEFORE the CTA so their session_state keys are always
    # populated when the click handler reads them (P1 fix: they used to live
    # below the results).
    render_settings_row()

    if uploaded_file is not None:
        if st.button("Transcribir", type="primary", use_container_width=True):
            model = st.session_state.get('model', 'Deepgram')
            # None = the provider auto-detects the language (Deepgram via
            # detect_language, Whisper natively).
            language_code = LANGUAGE_OPTIONS.get(st.session_state.get('language'))
            if model == "OpenAI Whisper" and not get_secret("OPENAI_API_KEY"):
                st.error("Falta la clave de OpenAI: añade `OPENAI_API_KEY` en los *secrets* o en un archivo `.env`.")
                return
            elif model == "Deepgram" and not get_secret("DEEPGRAM_API_KEY"):
                st.error("Falta la clave de Deepgram: añade `DEEPGRAM_API_KEY` en los *secrets* o en un archivo `.env`.")
                return
            status_box = st.status(f"Transcribiendo con {model}…", expanded=True)
            with status_box:
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
                            status_text.text("Convirtiendo M4A a MP3…")
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_tmp_file:
                                mp3_file_path = mp3_tmp_file.name
                            converted_path = convert_m4a_to_mp3(tmp_file_path, mp3_file_path)
                            temp_files_to_cleanup.append(converted_path)
                            audio_path = converted_path
                        elif file_extension == 'mp4':
                            status_text.text("Extrayendo el audio del MP4…")
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
                        status_text.text("Recortando el audio…")
                        if audio_path is not None:
                            trimmed_path = trim_audio_file(
                                audio_path,
                                trim_settings['start_time_ms'],
                                trim_settings['end_time_ms']
                            )
                            temp_files_to_cleanup.append(trimmed_path)
                            audio_path = trimmed_path
                    if audio_path is not None:
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
                                    status_text.text(progress_message_es(message))

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
                        st.error("Error interno: no hay ruta de audio que transcribir.")
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

                    progress_bar.progress(100)
                    st.session_state.transcription = transcription
                    st.session_state.analysis = analysis
                    st.session_state.filename = uploaded_file.name
                    status_box.update(label="Transcripción completada", state="complete", expanded=False)
                except Exception as e:
                    status_box.update(label="La transcripción falló", state="error", expanded=True)
                    st.error(f"Error durante la transcripción: {str(e)}")
    # Results (transcript + analysis tabs); persists across reruns until a new
    # transcription replaces it.
    if 'transcription' in st.session_state:
        render_results()

    st.markdown(
        '<hr style="margin:1.5rem 0 0.6rem 0;border:none;border-top:1px solid #E4E8E6;">'
        '<div class="vt-footer">MP3 · WAV · M4A · MP4 — creado por Juan Giraldo, '
        'con Streamlit, Deepgram y OpenAI<br>'
        '<a href="https://chatgpt.com/g/g-6874e87c48608191afa2da8e3e769279-generadoractasreunion" '
        'target="_blank">Genera el acta de la reunión con este GPT</a></div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main() 
