import streamlit as st
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import json
import shutil
import io
import logging
import imageio_ffmpeg
import subprocess

# Patch pydub to use imageio-ffmpeg's bundled ffmpeg/ffprobe BEFORE importing AudioSegment
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
print(f"Using FFmpeg from: {ffmpeg_path}")
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

# --- v2 visual system -------------------------------------------------------
# One CSS block, token-driven. Colors come from the theme in
# .streamlit/config.toml via Streamlit's CSS variables (with hex fallbacks), so
# the custom styles survive a user switching to the dark base theme. Layout
# stays mobile-first: 44px tap targets, >=16px inputs (stops iOS auto-zoom).
# Design rationale: docs/rediseno-ux-v2.md.
#
# Injected from main() (not at import) so every rerun/session gets the styles
# even when this module is imported by a wrapper script instead of being the
# Streamlit entry point itself.
def _inject_styles() -> None:
    st.markdown("""
<style>
    html { -webkit-text-size-adjust: 100%; }
    .block-container {
        max-width: 680px !important;
        margin-left: auto;
        margin-right: auto;
        padding-top: 4rem !important;
    }
    /* Dev chrome, not part of the product UI */
    .stAppDeployButton { display: none; }
    @media (max-width: 600px) {
        .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    /* Mobile-friendly tap targets */
    .stButton > button,
    .stDownloadButton > button,
    .stLinkButton > a {
        width: 100%;
        min-height: 44px;
    }
    div[data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {
        min-height: 44px;
        font-size: 16px !important;
    }
    .stFileUploader { width: 100%; }

    /* Hero: SVG mark + wordmark + tagline. The only centered block. */
    .vt-hero { text-align: center; margin: 0 0 1.4rem 0; }
    .vt-hero svg { display: block; margin: 0 auto 0.5rem auto; }
    .vt-hero svg rect { fill: var(--primary-color, #0F766E); }
    .vt-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .vt-tagline {
        font-size: 0.95rem;
        margin-top: 0.3rem;
        color: color-mix(in srgb, var(--text-color, #1F2A2E) 62%, transparent);
    }

    /* Section eyebrows carry the page structure (Audio / Ajustes / Resultado) */
    .vt-eyebrow {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.7rem 0 0.3rem 0;
        color: color-mix(in srgb, var(--text-color, #1F2A2E) 55%, transparent);
    }

    .vt-footer {
        text-align: center;
        font-size: 0.85rem;
        line-height: 1.8;
        color: color-mix(in srgb, var(--text-color, #1F2A2E) 62%, transparent);
    }
</style>
""", unsafe_allow_html=True)

# Core's progress messages ship with a leading emoji for the CLI; the app strips
# it at the presentation layer (core is shared and stays untouched).
_LEADING_SYMBOLS_RE = re.compile(r"^[^\w¿¡(]+", re.UNICODE)


def _clean_progress_message(message: str) -> str:
    return _LEADING_SYMBOLS_RE.sub("", message or "").strip()


def _eyebrow(label: str) -> None:
    """Uppercase section label — the page's structural device."""
    st.markdown(f'<div class="vt-eyebrow">{label}</div>', unsafe_allow_html=True)


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
            status_text.text("Transcribiendo…")
        return transcribe_with_openai(file_path, language)
    if status_text:
        status_text.text(f"El archivo se dividió en {len(chunk_paths)} fragmentos")
    transcriptions = []
    temp_dir = os.path.dirname(chunk_paths[0])
    try:
        for i, chunk_path in enumerate(chunk_paths, 1):
            if status_text:
                status_text.text(f"Procesando fragmento {i}/{len(chunk_paths)}…")
            if progress_bar:
                progress_bar.progress(i / len(chunk_paths))
            try:
                transcriptions.append(transcribe_with_openai(chunk_path, language))
                if status_text:
                    status_text.text(f"Fragmento {i} listo")
            except Exception as e:
                if status_text:
                    status_text.text(f"Error en el fragmento {i}: {str(e)}")
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
            raise Exception("La conversión de M4A no está disponible en este servidor. Convierte el archivo a MP3 o WAV antes de subirlo (hay un conversor en scripts/).")

        # Load the audio file
        audio = AudioSegment.from_file(str(input_file), format="m4a")

        # Export as MP3
        audio.export(str(output_file), format="mp3", bitrate=bitrate)

        logger.info(f"Successfully converted {input_path} to {output_file}")
        return str(output_file)

    except Exception as e:
        logger.error(f"Error converting {input_path}: {str(e)}")
        if "ffprobe" in str(e) or "ffmpeg" in str(e):
            raise Exception("La conversión de M4A no está disponible en este servidor. Convierte el archivo a MP3 o WAV antes de subirlo.")
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
        st.error("La sesión de Google expiró o no se pudo validar. Pulsa el botón de nuevo.")
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
        st.error(f"No se pudo completar el login de Google: {e}")
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
        # Secondary on purpose: the page's single primary action is Transcribir.
        if st.button("Cargar de Drive", key="drive_load_btn", use_container_width=True):
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


def _render_insights(insights: dict) -> None:
    """Metrics tab: evaluative summary first, per-speaker meters after, reading
    guide last. Speaker identity is always carried by text (never color alone),
    and the labels match the transcript's `[Speaker N]` names."""
    for item in insights["feedback"]:
        st.markdown(f"- {item}")
    st.divider()
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
    with st.expander("Cómo leer estas métricas"):
        for guide_line in METRIC_GUIDE_ES:
            st.markdown(f"- {guide_line}")


def _render_sentiment(sentiment: dict) -> None:
    """Sentiment tab. The 😊😐🙁 glyphs are data (the timeline's encoding), not
    decoration — the tone label reuses them as its legend."""
    avg = sentiment["average"]
    label_es = {"positive": "😊 Positivo", "neutral": "😐 Neutral",
                "negative": "🙁 Negativo"}.get(avg["sentiment"], avg["sentiment"])
    st.markdown(
        f"**Tono general:** {label_es} ({avg['sentiment_score']:+.2f}) — "
        f"{describe_sentiment_score(avg['sentiment_score'])}."
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
    st.caption(
        "Escala −1…+1 · Deepgram etiqueta positivo a partir de +0.33 "
        "y negativo por debajo de −0.33."
    )


def main():
    _inject_styles()

    # Complete any pending Google Drive OAuth redirect before rendering the UI.
    _handle_drive_oauth_callback()

    # Hero: waveform mark + wordmark + what-the-app-does tagline.
    st.markdown(
        '<div class="vt-hero">'
        '<svg width="34" height="30" viewBox="0 0 34 30" aria-hidden="true">'
        '<rect x="1" y="11" width="4" height="8" rx="2"/>'
        '<rect x="8" y="7" width="4" height="16" rx="2"/>'
        '<rect x="15" y="3" width="4" height="24" rx="2"/>'
        '<rect x="22" y="8" width="4" height="14" rx="2"/>'
        '<rect x="29" y="12" width="4" height="6" rx="2"/>'
        '</svg>'
        '<div class="vt-title">Voice Transcriber</div>'
        '<div class="vt-tagline">Transcripción con hablantes, métricas de conversación y sentimiento</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Audio ---------------------------------------------------------------
    _eyebrow("Audio")
    allowed_types = ['mp3', 'wav']
    if ffmpeg_available:
        allowed_types.append('m4a')
        allowed_types.append('mp4')

    # Two input sources: a local upload or a file from the user's Google Drive.
    # Both converge on `uploaded_file` (a real UploadedFile or a DriveFile);
    # everything downstream is source-agnostic.
    tab_local, tab_drive = st.tabs(["Archivo local", "Google Drive"])
    with tab_local:
        local_file = st.file_uploader(
            "Archivo de audio",
            type=allowed_types,
            help="MP3, WAV, M4A o MP4" if ffmpeg_available
                 else "MP3 o WAV (M4A/MP4 necesitan FFmpeg, no disponible en este servidor)",
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
                            raise Exception("La extracción de audio de MP4 requiere FFmpeg y no está disponible en este servidor.")
                        cmd = f"ffmpeg -i {tmp_file_path} -vn -acodec libmp3lame -ab 192k -ar 44100 -y {mp3_file_path}"
                        try:
                            subprocess.run(cmd, shell=True, check=True)
                            audio_path = mp3_file_path
                            st.session_state.converted_mp3_path = audio_path
                            st.session_state.converted_mp3_file_key = file_key
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Error extracting audio from MP4: {e}")
                            raise Exception(f"No se pudo extraer el audio del MP4: {e}")
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
                st.error(f"No se pudo leer el audio: {str(e)}")
                trim_settings = None
                return
        audio_info = st.session_state.audio_info
        duration_seconds = audio_info['duration_seconds']

        # Loaded-file summary: name · size · duration, one quiet line.
        st.caption(
            f"{uploaded_file.name} · {uploaded_file.size / 1024 / 1024:.1f} MB · "
            f"{format_time(duration_seconds)}"
        )

        # Trimming lives in a closed expander: most transcriptions take the
        # whole file, so the slider only appears when asked for.
        with st.expander("Recortar audio (opcional)"):
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
                "Rango a transcribir",
                min_value=0.0,
                max_value=max_seconds,
                step=0.1,
                format="%.1f s",
                key="trim_range_slider",
                on_change=_sync_from_slider,
            )

            # Precise entry (much easier than dragging on a phone): numeric start/end
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

            start_time, end_time = st.session_state.trim_range_slider

            # Show trim preview
            trim_duration = end_time - start_time
            st.caption(
                f"Se transcribirá {format_time(start_time)} – {format_time(end_time)} "
                f"({format_time(trim_duration)} en total)"
            )

        start_time, end_time = st.session_state.trim_range_slider

        # Store trim settings
        trim_settings = {
            'start_time_ms': int(start_time * 1000),
            'end_time_ms': int(end_time * 1000),
            'duration_ms': audio_info['duration_ms']
        }

    # --- Ajustes -------------------------------------------------------------
    # Only the language is a first-class control (the one knob casual users
    # touch); model + analysis toggles live behind "Opciones avanzadas" with
    # sensible defaults. Widgets render BEFORE the CTA that reads them.
    _eyebrow("Ajustes")
    st.segmented_control(
        "Idioma del audio",
        ["Auto", "Español", "English"],
        key='language',
        default="Auto",
        help="Con Auto, el proveedor detecta un idioma dominante él solo. "
             "Fijarlo afina algo la precisión en audios muy mezclados.",
    )
    with st.expander("Opciones avanzadas"):
        model = st.selectbox(
            "Modelo de transcripción",
            ["Deepgram", "OpenAI Whisper"],
            key='model',
            help="Deepgram (nova-3) añade hablantes, métricas y sentimiento. "
                 "Whisper es la alternativa de solo texto."
        )
        whisper_selected = model == "OpenAI Whisper"
        st.toggle(
            "Identificar hablantes (diarización)",
            key='diarize',
            value=True,
            disabled=whisper_selected,
            help="Marca quién dice qué con líneas [Speaker 0], [Speaker 1]… Solo con Deepgram."
        )
        st.toggle(
            "Métricas de conversación",
            key='insights',
            value=True,
            disabled=whisper_selected,
            help="Reparto del habla, ritmo (palabras/min), interrupciones, monólogos y "
                 "muletillas, con observaciones. Funciona en español e inglés."
        )
        st.toggle(
            "Análisis de sentimiento",
            key='sentiment',
            value=False,
            disabled=whisper_selected,
            help="Tono positivo/neutral/negativo por tramo y por hablante. Deepgram solo "
                 "lo ofrece para audio en inglés; en español se omite con un aviso."
        )

    # --- Transcribir ---------------------------------------------------------
    if uploaded_file is None:
        st.caption("Sube o carga un audio para empezar.")
    else:
        if st.button("Transcribir", type="primary", use_container_width=True):
            model = st.session_state.get('model', 'Deepgram')
            # Obtener idioma seleccionado (el segmented control devuelve None si
            # el usuario deselecciona: se trata como Auto).
            language_ui = st.session_state.get('language') or 'Auto'
            # Mapear a código de idioma; None = el proveedor detecta el idioma
            # (Deepgram vía detect_language, Whisper de forma nativa).
            if 'auto' in language_ui.lower():
                language_code = None
            elif 'espa' in language_ui.lower():
                language_code = 'es'
            else:
                language_code = 'en'
            if model == "OpenAI Whisper" and not get_secret("OPENAI_API_KEY"):
                st.error("Falta la clave OPENAI_API_KEY. Añádela en `.env` (local) o en los *secrets* (Streamlit Cloud).")
                return
            elif model == "Deepgram" and not get_secret("DEEPGRAM_API_KEY"):
                st.error("Falta la clave DEEPGRAM_API_KEY. Añádela en `.env` (local) o en los *secrets* (Streamlit Cloud).")
                return
            with st.status("Transcribiendo…", expanded=True) as status_box:
                progress_bar = st.progress(0.0)
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
                                status_text.text("Convirtiendo M4A a MP3…")
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_tmp_file:
                                mp3_file_path = mp3_tmp_file.name
                            converted_path = convert_m4a_to_mp3(tmp_file_path, mp3_file_path)
                            temp_files_to_cleanup.append(converted_path)
                            audio_path = converted_path
                        elif file_extension == 'mp4':
                            if status_text:
                                status_text.text("Extrayendo el audio del MP4…")
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_tmp_file:
                                mp3_file_path = mp3_tmp_file.name
                            # Use ffmpeg to extract audio from MP4
                            # This requires ffmpeg to be installed and in PATH
                            if not ffmpeg_available:
                                raise Exception("La extracción de audio de MP4 requiere FFmpeg y no está disponible en este servidor.")
                            cmd = f"ffmpeg -i {tmp_file_path} -vn -acodec libmp3lame -ab 192k -ar 44100 -y {mp3_file_path}"
                            try:
                                subprocess.run(cmd, shell=True, check=True)
                                audio_path = mp3_file_path
                                temp_files_to_cleanup.append(audio_path)
                            except subprocess.CalledProcessError as e:
                                logger.error(f"Error extracting audio from MP4: {e}")
                                raise Exception(f"No se pudo extraer el audio del MP4: {e}")
                            except Exception as e:
                                logger.error(f"Error during MP4 extraction: {e}")
                                raise e

                    # Apply trimming if settings are provided
                    # Note: audio_path is already MP3 at this point (either from cache or conversion)
                    if trim_settings and (trim_settings['start_time_ms'] > 0 or trim_settings['end_time_ms'] < trim_settings['duration_ms']):
                        if status_text:
                            status_text.text("Recortando el audio…")
                        if audio_path is not None:
                            trimmed_path = trim_audio_file(
                                audio_path,
                                trim_settings['start_time_ms'],
                                trim_settings['end_time_ms']
                            )
                            temp_files_to_cleanup.append(trimmed_path)
                            audio_path = trimmed_path
                    if audio_path is None:
                        raise Exception("Error interno: no se pudo preparar el audio.")
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
                                status_text.text(_clean_progress_message(message))

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
                    st.session_state.transcription = transcription
                    st.session_state.analysis = analysis
                    st.session_state.filename = uploaded_file.name
                    status_box.update(label="Transcripción completada", state="complete", expanded=False)
                except Exception as e:
                    status_box.update(label="La transcripción falló", state="error", expanded=True)
                    st.error(f"Error durante la transcripción: {str(e)}")

    # --- Resultado -----------------------------------------------------------
    if 'transcription' in st.session_state:
        _eyebrow("Resultado")

        analysis = st.session_state.get('analysis')
        meta_bits = [st.session_state.get('filename', '')]
        if analysis:
            if analysis.get("detected_language"):
                meta_bits.append(f"idioma: {analysis['detected_language']}")
            if analysis.get("model_used"):
                meta_bits.append(f"modelo: {analysis['model_used']}")
            if analysis.get("duration"):
                meta_bits.append(format_time(analysis['duration']))
        st.caption(" · ".join(bit for bit in meta_bits if bit))
        if analysis:
            for warning in analysis.get("warnings", []):
                st.caption(f"Aviso: {warning}")

        insights = analysis.get("insights") if analysis else None
        sentiment = analysis.get("sentiment") if analysis else None

        # Transcript + analysis as tabs (summary-first: the analysis tabs open
        # with the evaluative feedback, details after). With nothing but the
        # transcript (e.g. Whisper), skip the tab chrome entirely.
        tab_labels = (["Transcripción"]
                      + (["Métricas"] if insights else [])
                      + (["Sentimiento"] if sentiment else []))
        if len(tab_labels) == 1:
            with st.container(height=340):
                st.code(st.session_state.transcription, language=None, wrap_lines=True)
        else:
            tabs = st.tabs(tab_labels)
            with tabs[0]:
                # st.code keeps the built-in copy button; the fixed-height
                # container stops long transcripts from swallowing the page.
                with st.container(height=340):
                    st.code(st.session_state.transcription, language=None, wrap_lines=True)
            next_tab = 1
            if insights:
                with tabs[next_tab]:
                    _render_insights(insights)
                next_tab += 1
            if sentiment:
                with tabs[next_tab]:
                    _render_sentiment(sentiment)

        download_data = st.session_state.transcription.encode('utf-8')
        transcript_name = f"{st.session_state.filename.split('.')[0]}_transcript.txt"
        if analysis and (analysis.get("insights") or analysis.get("sentiment")):
            report_data = build_report(
                analysis, filename=st.session_state.get('filename', '')
            ).encode('utf-8')
            dl_txt, dl_report = st.columns(2)
            with dl_txt:
                st.download_button(
                    label="Descargar transcripción (.txt)",
                    data=download_data,
                    file_name=transcript_name,
                    mime="text/plain",
                    use_container_width=True,
                )
            with dl_report:
                st.download_button(
                    label="Descargar informe completo (.txt)",
                    data=report_data,
                    file_name=f"{st.session_state.filename.split('.')[0]}_report.txt",
                    mime="text/plain",
                    help="Transcripción más métricas de conversación y sentimiento en un solo archivo",
                    use_container_width=True,
                )
        else:
            st.download_button(
                label="Descargar transcripción (.txt)",
                data=download_data,
                file_name=transcript_name,
                mime="text/plain",
                use_container_width=True,
            )

    # --- Pie -----------------------------------------------------------------
    st.divider()
    st.markdown(
        '<div class="vt-footer">'
        'MP3 · WAV · M4A · MP4<br>'
        '<a href="https://chatgpt.com/g/g-6874e87c48608191afa2da8e3e769279-generadoractasreunion" target="_blank">Generar el acta de la reunión con este GPT</a><br>'
        'Hecho por Juan Giraldo · Streamlit + Deepgram + OpenAI'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
