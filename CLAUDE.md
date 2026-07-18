# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Streamlit voice-transcription web app. Users provide audio/video (MP3, WAV, M4A, MP4) either by **local upload** or from their **Google Drive** (via a `st.tabs` source selector), optionally trim it, and transcribe with either **Deepgram** (default) or **OpenAI** (gpt-4o-transcribe family) — both support speaker diarization. UI strings mix Spanish and English — Spanish/English are the two supported transcription languages. The UI is **mobile-first** (`layout="centered"`, full-width tap targets, numeric trim inputs alongside the slider).

## Product direction (read before proposing new features)

The shipped app is a **post-meeting file-upload transcriber**. The active goal is to evolve it into a live meeting-notes tool ("Granola propio") that produces structured **actas** (minutes), not just raw transcripts. Planning docs (in Spanish) define this and constrain how new work should be approached — read them before proposing feature work so you don't contradict decisions already made:

- [ROADMAP.md](ROADMAP.md) — the phased vision: Phase 0 extract shared transcription/diarization into a reusable module (it's currently copy-pasted, see below); Phase 1 add LLM-generated actas + templates + a "my notes" merge (highest ROI); Phase 2 in-browser live capture (Chrome MV3); Phase 3 desktop app for native Zoom/Teams.
- [docs/](docs/) — PRD, PO pitch, and a Lovable prompt playbook. **The currently favored path is building the feature inside an existing Lovable app (React + Supabase), not a Chrome extension.**

Hard constraints the docs establish, worth respecting in any suggestion: **API keys (Deepgram/LLM) must never live in client code** (extension or published app) — they require a backend/proxy; the **diarization-with-speaker-remapping logic is the crown-jewel asset** to preserve and reuse, not rewrite; and live capture needs Deepgram's **streaming WebSocket**, which behaves differently from the current batch endpoint (speaker remapping must be rethought for continuous flow). These docs are vision, not committed dates.

## Commands

```bash
# Run the web app (entry point)
py -m streamlit run AppTranscribe.py     # Windows
streamlit run AppTranscribe.py           # Linux/Mac

# Install deps
pip install -r requirements.txt

# CLI: Deepgram transcription with diarization (newest, preferred CLI)
python scripts/deepgram_transcribe_cli.py --file "C:\path\audio.mp4"
python scripts/deepgram_transcribe_cli.py --file audio.wav --language es --no-diarize
#   stdout = transcript, stderr = progress/metadata, writes output/<stem>_transcript.txt

# CLI: legacy OpenAI Whisper batch (processes input/ folder by default)
python scripts/transcribe.py --file audio.mp3
```

API keys go in `.env` (gitignored; see `.env.example`): `DEEPGRAM_API_KEY` and `OPENAI_API_KEY`. There is no test suite, linter, or build step.

## Architecture

The whole app lives in [AppTranscribe.py](AppTranscribe.py) (~1000 lines, single file). Key structure:

- **FFmpeg patching (lines 16-24) must run before `from pydub import AudioSegment` is used for real work.** The app bundles FFmpeg via `imageio-ffmpeg` and assigns its path to `AudioSegment.converter`/`AudioSegment.ffmpeg` so pydub works without a system FFmpeg on PATH. Do not reorder these imports. `check_ffmpeg()` sets a module-level `ffmpeg_available` flag — when false (no FFmpeg), M4A/MP4 upload types are hidden from the uploader.

- **Transcription pipeline** (bottom-up): `transcribe_with_openai` / `transcribe_with_deepgram` (single file) → `transcribe_file` (dispatch by model name string `"Deepgram"` / `"OpenAI"`) → `transcribe_large_file` (splits files >24MB into MP3 chunks via `split_audio_file`, then loops). The model is selected by a UI string, not an enum — match those exact strings.

- **Deepgram fallback chain**: `transcribe_with_deepgram` tries up to 3 endpoints in order — (1) explicit language + `model=base`, (2) `detect_language=true`, (3) plain endpoint. It accepts a result only if the transcript is longer than ~10 chars, otherwise falls through to the next attempt. `return_raw=True` returns the JSON dict (used by the diarization path).

- **OpenAI path**: non-diarized runs walk `OPENAI_TRANSCRIBE_MODELS` (`gpt-4o-transcribe` → `gpt-4o-mini-transcribe` → `whisper-1`) with the same >10-chars acceptance rule; diarized runs call `gpt-4o-transcribe-diarize` (`response_format="diarized_json"`, `chunking_strategy="auto"` always — required for >30s audio) via `transcribe_openai_diarized_raw`, and `openai_segments_to_deepgram_shape` adapts its string-labeled segments into the Deepgram utterance shape so the shared diarization helpers below are reused unchanged. Known-speaker params go through `extra_body` deliberately (SDK-version safety) — don't "clean up" into typed kwargs.

- **Diarization** is the most intricate logic. Both providers return per-utterance/segment speaker IDs that are **only consistent within a single API call** — so for chunked large files, speaker IDs must be remapped across chunks. `transcribe_large_file_with_diarization` (Deepgram) maintains a global speaker map, using `map_speakers_between_chunks` (matches speakers by frequency/last-speaker heuristics) and `format_diarized_output` (groups consecutive same-speaker utterances into `[Speaker N]: text` lines). `transcribe_large_file_with_diarization_openai` keeps cross-chunk consistency with a hybrid: it sends 2–10s reference clips of already-identified speakers (`extract_speaker_reference_clip`) as `known_speaker_names`/`known_speaker_references` so the API anchors them natively, and falls back to `map_speakers_between_chunks` for unmatched labels. If you change the diarization output format, change it consistently across these functions.

- **Format conversion + caching in `main()`**: M4A is converted to MP3 via pydub; MP4 audio is extracted via an `ffmpeg` subprocess (`-vn -acodec libmp3lame`). Conversions are **cached in `st.session_state`** (`converted_mp3_path` / `converted_mp3_file_key`, keyed by `name_size_type`) specifically to avoid re-converting on every Streamlit rerun (e.g. when the user moves the trim slider or clicks download). When touching upload/conversion/trim logic, preserve this cache — invalidating it incorrectly causes expensive re-conversions (this was the subject of recent bug fixes; see git log).

- **Trimming**: a range slider produces start/end ms; `trim_audio_file` slices the (already-MP3) audio. Trimming happens *after* conversion so it operates on MP3. The slider is mirrored by two `st.number_input` fields (Start/End seconds) for precise touch entry; the slider key and the two input keys are the source of truth, kept in sync via `on_change` callbacks (never bind two widgets to one key — Streamlit raises). Reset per file via `trim_state_key`.

- **Input sources (local upload OR Google Drive)**: the whole pipeline downstream of the uploader depends only on four members of one object — `.name`, `.size`, `.type`, `.getvalue()`. `DriveFile` is a duck-typed stand-in for Streamlit's `UploadedFile` exposing exactly those, so a Drive download flows through the cache/conversion/trim/transcription path unchanged. **The load-bearing invariant: `.name` must keep the real extension** (downstream routes conversion by `name.split('.')[-1]`); the Drive `fileId` is folded into `.type` only to keep `file_key` unique. The Drive helpers (`_build_drive_flow`, `_handle_drive_oauth_callback`, `list_drive_audio`, `download_drive_file`, `render_drive_tab`) live between `convert_m4a_to_mp3` and `main()`; google libs are imported lazily so the app runs without them (the Drive tab just shows a "not configured" note). `_handle_drive_oauth_callback()` runs first thing in `main()` and must `st.query_params.clear()` + `st.rerun()` after exchanging the code (a spent OAuth code fails on rerun).

  **OAuth + PKCE gotcha (do not "simplify" this):** when Google redirects back, the browser opens a **fresh Streamlit session**, so `st.session_state` from the login click is GONE in the callback. Anything that must survive the redirect (the PKCE `code_verifier`) cannot live in `session_state` — it would produce `invalid_grant: Missing code verifier`. Instead the verifier is persisted **on disk keyed by `state`** (`_pkce_save`/`_pkce_pop`, stored in `tempfile.gettempdir()/vt_oauth_pkce.json`); the callback reads `state` from the URL (which does survive) and looks the verifier up. The state-keyed lookup also serves as the CSRF guard (an attacker can't produce a `state` that indexes a saved verifier) and is single-use. The resulting `drive_creds` ARE stored in `session_state` — that's fine, because the callback runs in the same (redirect) session the user is now viewing.

- **Secrets**: read every API key/secret through `get_secret(key)` (near the top), which prefers `st.secrets` (Streamlit Cloud) and falls back to `os.environ`/`.env` (local). Do not reintroduce bare `os.environ.get(...)` for keys.

## scripts/ — standalone CLIs and converters

`scripts/` duplicates much of the app's transcription/diarization logic for command-line and batch use. **The diarization helpers (`format_diarized_output`, `map_speakers_between_chunks`, `extract_speakers_from_response`, `transcribe_large_file_with_diarization`) are copy-pasted between [AppTranscribe.py](AppTranscribe.py) and [scripts/deepgram_transcribe_cli.py](scripts/deepgram_transcribe_cli.py).** If you fix a diarization bug, check whether the same fix is needed in the other file. `deepgram_transcribe_cli.py` is the cleaner, newer reimplementation (uses `argparse.BooleanOptionalAction`, list-form subprocess calls, request timeouts); `transcribe.py` is the older OpenAI-only batch script. The `convert_*` scripts and `setup.*` scripts are self-contained utilities.

This copy-paste is exactly the debt Phase 0 of [ROADMAP.md](ROADMAP.md) wants gone — extract the shared transcription/diarization helpers into a single reusable module (proposed `core/transcription.py`) so future clients (extension, desktop, Lovable backend) reuse instead of copying. If you're already editing both copies to fix a bug, consider whether extracting is the better move.

## Deployment notes

This deploys to **Streamlit Cloud**. `packages.txt` (apt packages, contains `ffmpeg`) and `setup.sh` (`apt-get install ffmpeg`) exist to provide FFmpeg in that environment; `imageio-ffmpeg` is the primary mechanism, these are the fallback. Do not add `audioop-lts` to requirements — it was removed because it breaks the Python 3.11 deployment (see git history). The `.gitignore` excludes all audio files and `output/` — generated transcripts and test audio are never committed.

**Google Drive / OAuth setup** (optional feature). The Drive tab needs three secrets — `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI` — set in the Streamlit Cloud **Secrets** panel (and, for local dev, in `.env`; `get_secret` reads both). In Google Cloud Console: enable the **Google Drive API**, create an **OAuth client ID (Web application)**, and register **both** redirect URIs — `http://localhost:8501/` (local) and `https://<your-app>.streamlit.app/` (Cloud) — matching `GOOGLE_REDIRECT_URI` exactly (trailing slash included; a mismatch → `redirect_uri_mismatch` 400). The OAuth consent screen stays in **testing** mode (add yourself as a test user); note that in testing mode refresh tokens expire after ~7 days, so users periodically re-consent. `scope=drive.readonly` is intentional (least privilege for list+download). The three `google-*` libs in `requirements.txt` are pure-Python (no `packages.txt` change).
