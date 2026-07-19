"""Shared Deepgram transcription, diarization and conversation-analysis core.

This module is the single home for the logic that used to be copy-pasted
between AppTranscribe.py and scripts/deepgram_transcribe_cli.py (Phase 0 of
ROADMAP.md). It has NO Streamlit imports: clients pass the API key explicitly
and receive progress through an optional callback, so the same code can back
the Streamlit app, the CLI, and any future backend (extension / Lovable /
desktop).

Design notes (aligned with Deepgram's docs for the batch `/v1/listen` API):

- Model chain: requests are tried against a prioritized list of models
  (nova-3 / nova-2 first, legacy `base` last) instead of only `model=base`.
  Nova-3 supports Spanish monolingual (`language=es`) since 2025, and
  diarization accuracy on Nova-class models is significantly better.
- Diarization requests Deepgram's batch diarizer v2 (`diarize_model=latest`,
  GA May 2026 — the old `diarize=true` is deprecated and routes to v1), with
  automatic fallback to `diarize=true` if the param is rejected.
- Robust HTTP: every request has a timeout and transient failures
  (429 / 5xx / connection errors) are retried with exponential backoff.
- The fallback chain also applies in `return_raw` mode (the diarized path),
  which previously bypassed it entirely.
- Speaker IDs returned by Deepgram are only consistent WITHIN one API call.
  For chunked files the remapping is done with an injective (one-to-one)
  assignment based on talk-time share plus boundary continuity — the previous
  frequency-rank heuristic could map two current speakers onto the same
  previous speaker, silently merging voices.
- Chunking itself is minimized: Deepgram accepts far larger uploads than the
  24 MB limit inherited from Whisper, so most files are sent in ONE request,
  which makes speaker IDs consistent by construction. When chunking is still
  needed, split points are nudged to the quietest nearby moment so utterances
  don't straddle chunk boundaries.
- Audio intelligence: optional sentiment analysis (Deepgram `sentiment=true`,
  English-only per Deepgram docs) and locally computed conversation insights
  (talk share, pace, interruptions, muletillas/fillers, monologues) derived
  from diarized utterances — language-independent.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import time
from typing import Any, Callable

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

# Deepgram's batch endpoint accepts files up to 2 GB, so only chunk truly huge
# files. Keeping a file in ONE request keeps speaker IDs consistent without any
# remapping. (The old 24 MB threshold was inherited from Whisper's 25 MB cap.)
DEEPGRAM_MAX_CHUNK_MB = 150
# OpenAI Whisper still needs small chunks (25 MB API limit).
WHISPER_MAX_CHUNK_MB = 24

# Batch requests on long audio can take a while; Deepgram processes much
# faster than realtime, but leave generous room before giving up.
DEFAULT_TIMEOUT = 600  # seconds
MAX_HTTP_ATTEMPTS = 3  # per endpoint variant (429/5xx/connection retries)

# A transcript shorter than this is treated as a failed attempt so the next
# fallback endpoint gets a chance (same threshold the app has always used).
MIN_ACCEPTABLE_CHARS = 10

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
}

# Audio-intelligence features that Deepgram only offers for English input
# (per Deepgram docs: sentiment / intents / topics / summarize / filler_words
# are English-only). Requests for other languages must not include them.
ENGLISH_ONLY_FEATURES = {"sentiment", "intents", "topics", "summarize", "filler_words"}
# Features only available on nova-class models (rejected on base/enhanced).
NOVA_ONLY_FEATURES = {"filler_words"}


class DeepgramError(RuntimeError):
    """Raised when Deepgram could not produce a usable transcription."""


ProgressCb = Callable[[float | None, str], None]


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def candidate_models(language: str | None) -> list[dict[str, str]]:
    """Ordered list of (model, language-params) attempts for a target language.

    Newest models first (nova-3 supports both `language=es` and `language=en`
    monolingual, per the 2025 expansion); the legacy `base` model the app
    historically used is kept as a safety net, and `detect_language` closes
    the chain (Deepgram auto-falls back to the best model for a detected
    language). `language='multi'` targets nova-3 code-switching (mixed ES/EN).
    """
    if language == "es":
        return [
            {"model": "nova-3", "language": "es"},
            {"model": "nova-2", "language": "es"},
            {"model": "base", "language": "es"},
            {"model": "nova-2", "detect_language": "true"},
        ]
    if language == "en":
        return [
            {"model": "nova-3", "language": "en"},
            {"model": "nova-2", "language": "en"},
            {"model": "base", "language": "en"},
            {"model": "nova-2", "detect_language": "true"},
        ]
    if language == "multi":
        return [
            {"model": "nova-3", "language": "multi"},
            {"model": "nova-2", "detect_language": "true"},
            {"model": "base", "detect_language": "true"},
        ]
    # Unknown/unspecified language: let Deepgram detect it.
    return [
        {"model": "nova-3", "detect_language": "true"},
        {"model": "nova-2", "detect_language": "true"},
        {"model": "base", "detect_language": "true"},
        {},  # bare endpoint, Deepgram defaults
    ]


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

def _content_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _CONTENT_TYPES.get(ext, "application/octet-stream")


def _request_with_retries(
    params: dict[str, str],
    headers: dict[str, str],
    audio_data: bytes,
    timeout: float,
) -> requests.Response:
    """POST to Deepgram, retrying transient failures with backoff.

    Returns the final Response (which may still be a non-retryable 4xx —
    the caller decides what to do with it). Raises DeepgramError only when
    every attempt failed at the network level.
    """
    last_exc: Exception | None = None
    response: requests.Response | None = None
    for attempt in range(MAX_HTTP_ATTEMPTS):
        try:
            response = requests.post(
                DEEPGRAM_URL, params=params, headers=headers, data=audio_data,
                timeout=timeout,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            response = None
        if response is not None and response.status_code not in _RETRYABLE_STATUS:
            return response
        if attempt < MAX_HTTP_ATTEMPTS - 1:
            # Honor Retry-After on 429 when present, else exponential backoff.
            delay = 2.0 * (2 ** attempt)
            if response is not None:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
            time.sleep(min(delay, 30.0))
    if response is not None:
        return response
    raise DeepgramError(f"Could not reach Deepgram: {last_exc}")


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def get_transcript(raw: dict) -> str:
    """Plain transcript from a Deepgram response (safe on malformed input)."""
    try:
        channels = raw.get("results", {}).get("channels") or [{}]
        alternatives = channels[0].get("alternatives") or [{}]
        return alternatives[0].get("transcript", "") or ""
    except (AttributeError, IndexError, TypeError):
        return ""


def get_words(raw: dict) -> list[dict]:
    """Flat word list (with per-word speaker when diarized)."""
    try:
        channels = raw.get("results", {}).get("channels") or [{}]
        alternatives = channels[0].get("alternatives") or [{}]
        return alternatives[0].get("words") or []
    except (AttributeError, IndexError, TypeError):
        return []


def get_detected_language(raw: dict) -> str | None:
    try:
        channels = raw.get("results", {}).get("channels") or [{}]
        return channels[0].get("detected_language")
    except (AttributeError, IndexError, TypeError):
        return None


def get_model_used(raw: dict) -> str | None:
    try:
        info = raw.get("metadata", {}).get("model_info") or {}
        for model_meta in info.values():
            name = model_meta.get("name")
            if name:
                return name
    except (AttributeError, TypeError):
        pass
    return None


def get_audio_duration(raw: dict) -> float | None:
    try:
        duration = raw.get("metadata", {}).get("duration")
        return float(duration) if duration is not None else None
    except (TypeError, ValueError):
        return None


def _synthesize_utterances_from_words(words: list[dict]) -> list[dict]:
    """Rebuild utterance-like segments from the word list.

    Safety net for responses that carry per-word speakers but no
    `results.utterances` (e.g. if a fallback endpoint dropped the param):
    group consecutive words by speaker so diarized formatting still works.
    """
    utterances: list[dict] = []
    current: dict | None = None
    for word in words:
        speaker = word.get("speaker", 0)
        text = word.get("punctuated_word") or word.get("word") or ""
        if current is not None and current["speaker"] == speaker:
            current["transcript"] += (" " + text) if text else ""
            current["end"] = word.get("end", current["end"])
            current["words"].append(word)
        else:
            if current is not None:
                utterances.append(current)
            current = {
                "speaker": speaker,
                "transcript": text,
                "start": word.get("start", 0.0),
                "end": word.get("end", 0.0),
                "words": [word],
            }
    if current is not None:
        utterances.append(current)
    return utterances


def get_utterances(raw: dict) -> list[dict]:
    """Utterances from the response, synthesized from words if missing."""
    utterances = raw.get("results", {}).get("utterances") or []
    if utterances:
        return utterances
    words = get_words(raw)
    if words and any("speaker" in w for w in words):
        return _synthesize_utterances_from_words(words)
    return []


def _acceptable(raw: dict, diarize: bool) -> bool:
    """Quality gate deciding whether a response is good enough to accept."""
    if diarize and get_utterances(raw):
        return True
    return len(get_transcript(raw).strip()) > MIN_ACCEPTABLE_CHARS


# ---------------------------------------------------------------------------
# Core single-file transcription with fallback chain
# ---------------------------------------------------------------------------

def transcribe_with_deepgram(
    path: str,
    api_key: str,
    language: str | None = None,
    diarize: bool = False,
    return_raw: bool = False,
    features: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> str | dict:
    """Transcribe one audio file with Deepgram, trying models newest-first.

    Args:
        path: audio file path.
        api_key: Deepgram API key (callers own secret lookup).
        language: 'es', 'en' or None (auto-detect).
        diarize: request speaker labels (+utterances).
        return_raw: return the raw JSON response dict instead of text.
        features: extra query params, e.g. {"sentiment": "true"}. English-only
            features are stripped automatically for non-English requests, and
            stripped + retried if Deepgram rejects the combination.
        timeout: per-request timeout in seconds.

    Returns:
        Formatted transcript string, or the raw response dict when
        return_raw=True. Raises DeepgramError if every attempt failed outright.
    """
    if not api_key:
        raise EnvironmentError("DEEPGRAM_API_KEY environment variable not set")

    with open(path, "rb") as f:
        audio_data = f.read()

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": _content_type_for(path),
        "Accept": "application/json",
    }

    base_params: dict[str, str] = {"smart_format": "true", "punctuate": "true"}
    if diarize:
        # Batch diarizer v2 (`diarize_model=latest`, GA May 2026). The old
        # `diarize=true` is deprecated and routes to the v1 diarizer; if the
        # new param is ever rejected we fall back to it below.
        base_params["diarize_model"] = "latest"
        base_params["utterances"] = "true"

    errors: list[str] = []
    best_raw: dict | None = None  # weak-but-valid response kept as last resort

    for model_params in candidate_models(language):
        params = dict(base_params)
        params.update(model_params)

        extra = dict(features or {})
        # English-only intelligence features must not ride along on non-English
        # requests — Deepgram rejects the combination.
        request_lang = model_params.get("language")
        if request_lang not in (None, "en", "en-US", "en-GB"):
            extra = {k: v for k, v in extra.items() if k not in ENGLISH_ONLY_FEATURES}
        # filler_words only exists on nova-class models.
        if not model_params.get("model", "").startswith("nova"):
            extra = {k: v for k, v in extra.items() if k not in NOVA_ONLY_FEATURES}
        params.update(extra)

        response = _request_with_retries(params, headers, audio_data, timeout)

        # Unsupported model/feature/language combos come back as 400s. Degrade
        # gracefully before abandoning this model: first drop the optional
        # intelligence features, then swap the v2 diarizer param for the
        # deprecated-but-universal `diarize=true`.
        if response.status_code == 400 and extra:
            params = dict(base_params)
            params.update(model_params)
            response = _request_with_retries(params, headers, audio_data, timeout)
        if response.status_code == 400 and "diarize_model" in params:
            params = {k: v for k, v in params.items() if k != "diarize_model"}
            params["diarize"] = "true"
            response = _request_with_retries(params, headers, audio_data, timeout)

        if not response.ok:
            errors.append(
                f"{model_params.get('model', 'default')}: HTTP {response.status_code} {response.text[:200]}"
            )
            continue

        try:
            raw = response.json()
        except ValueError:
            errors.append(f"{model_params.get('model', 'default')}: invalid JSON response")
            continue

        if _acceptable(raw, diarize):
            if return_raw:
                return raw
            if diarize:
                return format_diarized_output(raw)
            return get_transcript(raw)

        # Keep the longest transcript seen so far in case nothing passes.
        if best_raw is None or len(get_transcript(raw)) > len(get_transcript(best_raw)):
            best_raw = raw

    if best_raw is not None:
        if return_raw:
            return best_raw
        return format_diarized_output(best_raw) if diarize else get_transcript(best_raw)

    message = "Deepgram failed to transcribe the audio. " + "; ".join(errors[-3:])
    if return_raw:
        return {"error": message}
    raise DeepgramError(message)


# ---------------------------------------------------------------------------
# Diarization formatting and cross-chunk speaker mapping
# ---------------------------------------------------------------------------

def format_diarized_output(deepgram_response: dict, speaker_mapping: dict | None = None) -> str:
    """Format utterances into `[Speaker N]: text` lines.

    Groups consecutive utterances of the same speaker. This output format is
    shared by the app and the CLI — change it here and nowhere else.
    """
    utterances = get_utterances(deepgram_response)
    if not utterances:
        return get_transcript(deepgram_response)

    formatted_lines: list[str] = []
    current_speaker: int | None = None
    current_text_parts: list[str] = []

    for utterance in utterances:
        speaker = utterance.get("speaker", 0)
        if speaker_mapping and speaker in speaker_mapping:
            speaker = speaker_mapping[speaker]

        transcript = (utterance.get("transcript") or "").strip()
        if not transcript:
            continue

        if speaker == current_speaker:
            current_text_parts.append(transcript)
        else:
            if current_speaker is not None and current_text_parts:
                formatted_lines.append(
                    f"[Speaker {current_speaker}]: {' '.join(current_text_parts)}"
                )
            current_speaker = speaker
            current_text_parts = [transcript]

    if current_speaker is not None and current_text_parts:
        formatted_lines.append(f"[Speaker {current_speaker}]: {' '.join(current_text_parts)}")

    return "\n".join(formatted_lines)


def extract_speakers_from_response(deepgram_response: dict) -> tuple[list[dict], dict]:
    """Utterances plus per-speaker stats: utterance count, talk time, words.

    Returns (utterances, {speaker_id: {"count": int, "duration": float,
    "words": int}}). Talk time is the strongest cross-chunk matching signal —
    utterance counts alone over-weight brief backchannels ("mm", "ok").
    """
    utterances = get_utterances(deepgram_response)
    if not utterances:
        return [], {}

    speaker_stats: dict[int, dict[str, float]] = {}
    for utterance in utterances:
        speaker = utterance.get("speaker", 0)
        stats = speaker_stats.setdefault(speaker, {"count": 0, "duration": 0.0, "words": 0})
        stats["count"] += 1
        try:
            stats["duration"] += max(0.0, float(utterance.get("end", 0.0)) - float(utterance.get("start", 0.0)))
        except (TypeError, ValueError):
            pass
        stats["words"] += len((utterance.get("transcript") or "").split())

    return utterances, speaker_stats


def _normalize_stats(stats: dict) -> dict[int, dict[str, float]]:
    """Accept both the new dict-valued stats and the legacy int counts."""
    normalized = {}
    for speaker, value in stats.items():
        if isinstance(value, dict):
            normalized[speaker] = {
                "count": value.get("count", 0),
                "duration": float(value.get("duration", 0.0) or 0.0),
                "words": value.get("words", 0),
            }
        else:  # legacy: plain utterance count
            normalized[speaker] = {"count": value, "duration": float(value), "words": 0}
    return normalized


def _talk_shares(stats: dict[int, dict[str, float]]) -> dict[int, float]:
    total = sum(s["duration"] for s in stats.values())
    if total <= 0:  # fall back to utterance counts when durations are missing
        total = sum(s["count"] for s in stats.values()) or 1.0
        return {k: s["count"] / total for k, s in stats.items()}
    return {k: s["duration"] / total for k, s in stats.items()}


def map_speakers_between_chunks(
    prev_speakers: dict,
    current_speakers: dict,
    prev_last_speaker: int | None = None,
    current_first_speaker: int | None = None,
) -> dict:
    """Map current-chunk speaker IDs onto previous-chunk speaker IDs.

    Injective by construction: each previous speaker is claimed by at most ONE
    current speaker (the old rank-based heuristic could assign two current
    speakers to the same previous ID, merging two voices into one). Scoring
    combines talk-time share similarity with a boundary-continuity bonus (the
    voice heard first in this chunk is likely the voice heard last in the
    previous one). Unmatched current speakers get fresh IDs.
    """
    if not prev_speakers or not current_speakers:
        return {}

    prev_stats = _normalize_stats(prev_speakers)
    cur_stats = _normalize_stats(current_speakers)
    prev_shares = _talk_shares(prev_stats)
    cur_shares = _talk_shares(cur_stats)

    scored: list[tuple[float, int, int]] = []
    for cur_id, cur_share in cur_shares.items():
        for prev_id, prev_share in prev_shares.items():
            score = 1.0 - abs(cur_share - prev_share)
            if (
                current_first_speaker is not None
                and prev_last_speaker is not None
                and cur_id == current_first_speaker
                and prev_id == prev_last_speaker
            ):
                score += 0.5
            scored.append((score, cur_id, prev_id))

    # Greedy one-to-one assignment, best matches first. Ties broken by lower
    # IDs to keep the mapping deterministic.
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    mapping: dict[int, int] = {}
    used_prev: set = set()
    for score, cur_id, prev_id in scored:
        if cur_id in mapping or prev_id in used_prev:
            continue
        mapping[cur_id] = prev_id
        used_prev.add(prev_id)

    # Extra speakers that appeared in this chunk get brand-new IDs.
    next_new = max(prev_stats.keys(), default=-1) + 1
    for cur_id in sorted(cur_stats.keys()):
        if cur_id not in mapping:
            mapping[cur_id] = next_new
            next_new += 1

    return mapping


def _normalize_utterances(
    utterances: list[dict],
    offset_s: float = 0.0,
    speaker_mapping: dict | None = None,
) -> list[dict]:
    """Copy utterances applying a time offset and a speaker mapping."""
    normalized = []
    for utterance in utterances:
        speaker = utterance.get("speaker", 0)
        if speaker_mapping and speaker in speaker_mapping:
            speaker = speaker_mapping[speaker]
        try:
            start = float(utterance.get("start", 0.0)) + offset_s
            end = float(utterance.get("end", 0.0)) + offset_s
        except (TypeError, ValueError):
            start, end = offset_s, offset_s
        normalized.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "transcript": (utterance.get("transcript") or "").strip(),
                "words": utterance.get("words") or [],
            }
        )
    return normalized


# ---------------------------------------------------------------------------
# Audio splitting (lazy pydub so this module imports without audio deps)
# ---------------------------------------------------------------------------

def _find_quiet_split_ms(audio, target_ms: int, window_ms: int = 10000, probe_ms: int = 400) -> int:
    """Pick the quietest instant near `target_ms` to use as a chunk boundary.

    Splitting at silence keeps utterances from straddling chunks, which is
    what makes cross-chunk speaker remapping fragile.
    """
    start = max(0, target_ms - window_ms)
    end = min(len(audio), target_ms + window_ms)
    if end - start <= probe_ms:
        return target_ms
    quietest_pos = target_ms
    quietest_rms = None
    for pos in range(start, end - probe_ms, probe_ms // 2):
        rms = audio[pos: pos + probe_ms].rms
        if quietest_rms is None or rms < quietest_rms:
            quietest_rms = rms
            quietest_pos = pos + probe_ms // 2
    return quietest_pos


def split_audio_file(file_path: str, max_size_mb: int = DEEPGRAM_MAX_CHUNK_MB) -> list[str]:
    """Split an oversized audio file into MP3 chunks (silence-aware cuts).

    Returns [file_path] untouched when the file already fits. Chunks are
    written to a fresh temp dir the caller must clean up (its path is the
    dirname of the returned chunks).
    """
    paths, _durations, _temp_dir = _split_audio_with_durations(file_path, max_size_mb)
    return paths


def _split_audio_with_durations(
    file_path: str, max_size_mb: int
) -> tuple[list[str], list[float], str | None]:
    """Like split_audio_file but also returns chunk durations (seconds)."""
    max_size_bytes = max_size_mb * 1024 * 1024
    file_size = os.path.getsize(file_path)
    if file_size <= max_size_bytes:
        return [file_path], [], None

    from pydub import AudioSegment  # lazy: only needed when actually splitting

    audio = AudioSegment.from_file(file_path)
    num_chunks = math.ceil(file_size / max_size_bytes)
    chunk_duration = len(audio) // num_chunks

    temp_dir = tempfile.mkdtemp(prefix="deepgram_chunks_")
    chunk_paths: list[str] = []
    durations: list[float] = []
    try:
        boundaries = [0]
        for i in range(1, num_chunks):
            target = i * chunk_duration
            boundaries.append(_find_quiet_split_ms(audio, target))
        boundaries.append(len(audio))
        # Guard against out-of-order boundaries from the silence search.
        for i in range(1, len(boundaries)):
            boundaries[i] = max(boundaries[i], boundaries[i - 1])

        for i in range(num_chunks):
            start_ms, end_ms = boundaries[i], boundaries[i + 1]
            chunk = audio[start_ms:end_ms]
            chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.mp3")
            chunk.export(chunk_path, format="mp3")
            chunk_paths.append(chunk_path)
            durations.append((end_ms - start_ms) / 1000.0)
        return chunk_paths, durations, temp_dir
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# Sentiment (Deepgram audio intelligence)
# ---------------------------------------------------------------------------

def _sentiment_label(score: float) -> str:
    # Mirrors Deepgram's segment labeling convention on the -1..1 score.
    if score >= 0.33:
        return "positive"
    if score <= -0.33:
        return "negative"
    return "neutral"


def extract_sentiment_points(
    raw: dict,
    offset_s: float = 0.0,
    speaker_mapping: dict | None = None,
) -> list[dict]:
    """Flatten `results.sentiments.segments` into per-word scored points.

    Each point: {"time": absolute seconds, "score": float, "speaker": global id}.
    Word indices in segments refer to the flat channel word list, whose words
    carry per-word speakers when diarization was on — that is what lets us
    attribute sentiment to speakers.
    """
    sentiments = raw.get("results", {}).get("sentiments") or {}
    segments = sentiments.get("segments") or []
    words = get_words(raw)
    if not segments or not words:
        return []

    points: list[dict] = []
    for segment in segments:
        try:
            start_word = int(segment.get("start_word", 0))
            end_word = int(segment.get("end_word", start_word))
            score = float(segment.get("sentiment_score", 0.0))
        except (TypeError, ValueError):
            continue
        for word in words[start_word: end_word + 1]:
            speaker = word.get("speaker", 0)
            if speaker_mapping and speaker in speaker_mapping:
                speaker = speaker_mapping[speaker]
            try:
                w_time = float(word.get("start", 0.0)) + offset_s
            except (TypeError, ValueError):
                w_time = offset_s
            points.append({"time": w_time, "score": score, "speaker": speaker})
    return points


def summarize_sentiment(points: list[dict], n_buckets: int = 12) -> dict | None:
    """Aggregate word-level sentiment points into an overall/per-speaker/timeline view."""
    if not points:
        return None

    total = sum(p["score"] for p in points)
    average = total / len(points)

    by_speaker: dict[Any, list[float]] = {}
    for p in points:
        by_speaker.setdefault(p["speaker"], []).append(p["score"])
    per_speaker = {
        speaker: {
            "sentiment_score": sum(scores) / len(scores),
            "sentiment": _sentiment_label(sum(scores) / len(scores)),
            "words": len(scores),
        }
        for speaker, scores in by_speaker.items()
    }

    t_min = min(p["time"] for p in points)
    t_max = max(p["time"] for p in points)
    span = max(t_max - t_min, 1e-6)
    buckets: list[list[float]] = [[] for _ in range(n_buckets)]
    for p in points:
        idx = min(int((p["time"] - t_min) / span * n_buckets), n_buckets - 1)
        buckets[idx].append(p["score"])
    timeline = []
    for i, bucket in enumerate(buckets):
        timeline.append(
            {
                "start": t_min + span * i / n_buckets,
                "end": t_min + span * (i + 1) / n_buckets,
                "score": (sum(bucket) / len(bucket)) if bucket else None,
            }
        )

    return {
        "average": {"sentiment_score": average, "sentiment": _sentiment_label(average)},
        "per_speaker": per_speaker,
        "timeline": timeline,
    }


# ---------------------------------------------------------------------------
# Conversation insights ("speech feedback") — computed locally from utterances
# ---------------------------------------------------------------------------

# Conservative filler/muletilla lexicons. Multi-word phrases are matched on the
# lowercased utterance text with word boundaries; thresholds below keep normal
# use of these words from being flagged.
FILLERS_ES = [
    "eh", "ehh", "em", "mmm", "este", "pues", "bueno", "vale", "o sea", "osea",
    "digamos", "es que", "en plan", "¿no?", "¿vale?", "¿sabes?", "ya sabes",
]
FILLERS_EN = [
    "uh", "um", "uhm", "mhmm", "mm-hmm", "hmm", "like", "you know", "i mean",
    "kind of", "sort of", "basically", "actually", "right?",
]

_MONOLOGUE_GAP_S = 2.0       # gaps shorter than this don't break a monologue
_INTERRUPTION_OVERLAP_S = 0.15

# --- Benchmarks -------------------------------------------------------------
# Orientative communication-coaching ranges used to qualify the raw numbers,
# so the UI/report can say "good/ok/high" instead of leaving values bare.

RATING_ICONS = {"good": "✅", "ok": "🟡", "warn": "⚠️", "info": "ℹ️"}

METRIC_GUIDE_ES = [
    "Ritmo: 120–160 palabras/min es el rango conversacional cómodo; <100 suena muy pausado y >180 acelerado.",
    "Silencio: 10–25% de pausas es natural al hablar; bastante más sugiere dudas o cortes, bastante menos, atropello.",
    "Muletillas: hasta ~3 por cada 100 palabras pasan desapercibidas; >5 empiezan a distraer.",
    "Reparto del habla: en una conversación equilibrada nadie supera ~55–60% del tiempo.",
    "Interrupciones: <1 por minuto es fluido; más indica solapamiento constante.",
    "Sentimiento (Deepgram): escala −1…+1; ≥ +0.33 se etiqueta positivo y ≤ −0.33 negativo; entre medias, neutral.",
    "Son rangos orientativos de coaching de comunicación — el contexto (entrevista, daily, venta) manda.",
]


def _rate_pace(wpm: float | None) -> dict | None:
    """Qualify words-per-minute against the conversational 120–160 band."""
    if wpm is None:
        return None
    if 120 <= wpm <= 160:
        return {"level": "good", "note": f"ritmo conversacional ideal ({wpm:.0f} ppm; rango 120–160)"}
    if 100 <= wpm < 120:
        return {"level": "ok", "note": f"ritmo algo pausado ({wpm:.0f} ppm; ideal 120–160)"}
    if 160 < wpm <= 180:
        return {"level": "ok", "note": f"ritmo algo rápido ({wpm:.0f} ppm; ideal 120–160)"}
    if wpm < 100:
        return {"level": "warn", "note": f"ritmo muy pausado ({wpm:.0f} ppm; ideal 120–160)"}
    return {"level": "warn", "note": f"ritmo muy rápido ({wpm:.0f} ppm; ideal 120–160)"}


def _rate_fillers(per_100: float, total: int) -> dict:
    if total == 0:
        return {"level": "good", "note": "sin muletillas detectadas"}
    if per_100 < 1.0:
        return {"level": "good", "note": f"muletillas muy contenidas ({per_100:.1f} por 100 palabras)"}
    if per_100 <= 3.0:
        return {"level": "ok", "note": f"muletillas en rango normal ({per_100:.1f} por 100 palabras; ideal <3)"}
    if per_100 <= 5.0:
        return {"level": "warn", "note": f"muletillas notorias ({per_100:.1f} por 100 palabras; ideal <3)"}
    return {"level": "warn", "note": f"muletillas muy frecuentes ({per_100:.1f} por 100 palabras; ideal <3)"}


def _rate_silence(ratio: float) -> dict:
    pct = ratio * 100
    if ratio < 0.05:
        return {"level": "ok", "note": f"casi sin pausas ({pct:.0f}% de silencio; típico 10–25%)"}
    if ratio <= 0.25:
        return {"level": "good", "note": f"pausas naturales ({pct:.0f}% de silencio; típico 10–25%)"}
    if ratio <= 0.40:
        return {"level": "ok", "note": f"bastantes pausas ({pct:.0f}% de silencio; típico 10–25%)"}
    return {"level": "warn", "note": f"mucho silencio ({pct:.0f}%; típico 10–25% — ¿dudas, cortes o espera?)"}


def describe_sentiment_score(score: float) -> str:
    """Spanish nuance for a Deepgram sentiment score (−1…+1, labels at ±0.33)."""
    if score >= 0.5:
        return "claramente positivo"
    if score >= 0.33:
        return "positivo"
    if score >= 0.1:
        return "neutral, tirando a positivo"
    if score > -0.1:
        return "neutro"
    if score > -0.33:
        return "neutral, tirando a negativo"
    if score > -0.5:
        return "negativo"
    return "claramente negativo"


def _count_fillers(text: str, lexicon: list[str]) -> dict[str, int]:
    lowered = f" {text.lower()} "
    counts: dict[str, int] = {}
    for filler in lexicon:
        if filler.endswith("?") or filler.startswith("¿"):
            n = lowered.count(filler)
        else:
            n = len(re.findall(rf"(?<![\wáéíóúüñ]){re.escape(filler)}(?![\wáéíóúüñ])", lowered))
        if n:
            counts[filler] = n
    return counts


def _merged_turns(utterances: list[dict]) -> list[dict]:
    """Merge consecutive same-speaker utterances (tolerating short gaps) into turns."""
    turns: list[dict] = []
    for u in utterances:
        if (
            turns
            and turns[-1]["speaker"] == u["speaker"]
            and u["start"] - turns[-1]["end"] <= _MONOLOGUE_GAP_S
        ):
            turns[-1]["end"] = max(turns[-1]["end"], u["end"])
            turns[-1]["transcript"] += " " + u["transcript"]
        else:
            turns.append(
                {
                    "speaker": u["speaker"],
                    "start": u["start"],
                    "end": u["end"],
                    "transcript": u["transcript"],
                }
            )
    return turns


def _fmt_mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def compute_speech_insights(
    utterances: list[dict],
    language: str | None = None,
    total_duration: float | None = None,
) -> dict | None:
    """Per-speaker conversation metrics + human-readable feedback (Spanish).

    Works on normalized utterances (absolute times, global speaker ids). All
    metrics are language-independent except the filler lexicon, chosen by
    `language` ('es' default, 'en' switches to English fillers).
    """
    utterances = [u for u in utterances if u.get("transcript")]
    if not utterances:
        return None

    utterances = sorted(utterances, key=lambda u: (u["start"], u["end"]))
    lexicon = FILLERS_EN if language == "en" else FILLERS_ES

    t_start = utterances[0]["start"]
    t_end = max(u["end"] for u in utterances)
    span = max(t_end - t_start, 1e-6)
    if total_duration and total_duration > span:
        span = total_duration

    per_speaker: dict[Any, dict] = {}
    for u in utterances:
        s = per_speaker.setdefault(
            u["speaker"],
            {
                "talk_time": 0.0, "words": 0, "turns": 0, "questions": 0,
                "fillers": {}, "longest_monologue": 0.0, "interruptions_made": 0,
            },
        )
        s["talk_time"] += max(0.0, u["end"] - u["start"])
        s["words"] += len(u["transcript"].split())
        s["questions"] += u["transcript"].count("?")
        for filler, n in _count_fillers(u["transcript"], lexicon).items():
            s["fillers"][filler] = s["fillers"].get(filler, 0) + n

    turns = _merged_turns(utterances)
    for turn in turns:
        s = per_speaker[turn["speaker"]]
        s["turns"] += 1
        s["longest_monologue"] = max(s["longest_monologue"], turn["end"] - turn["start"])

    # Interruptions: a different speaker starts while the previous utterance
    # is still running (attributed to the one who cut in).
    interruptions_total = 0
    for prev, cur in zip(utterances, utterances[1:]):
        if cur["speaker"] != prev["speaker"] and cur["start"] < prev["end"] - _INTERRUPTION_OVERLAP_S:
            per_speaker[cur["speaker"]]["interruptions_made"] += 1
            interruptions_total += 1

    total_talk = sum(s["talk_time"] for s in per_speaker.values()) or 1e-6
    for speaker, s in per_speaker.items():
        s["talk_share"] = s["talk_time"] / total_talk
        s["wpm"] = (s["words"] / (s["talk_time"] / 60.0)) if s["talk_time"] >= 5.0 else None
        s["fillers_total"] = sum(s["fillers"].values())
        s["fillers_per_100_words"] = (
            100.0 * s["fillers_total"] / s["words"] if s["words"] else 0.0
        )
        s["pace_rating"] = _rate_pace(s["wpm"])
        s["fillers_rating"] = _rate_fillers(s["fillers_per_100_words"], s["fillers_total"])

    # Voiced time union (utterances can overlap during cross-talk).
    voiced = 0.0
    cursor = None
    for u in utterances:
        s0, e0 = u["start"], u["end"]
        if cursor is None or s0 > cursor:
            voiced += max(0.0, e0 - s0)
            cursor = e0
        elif e0 > cursor:
            voiced += e0 - cursor
            cursor = e0
    silence_ratio = max(0.0, 1.0 - voiced / span)

    total_words = sum(s["words"] for s in per_speaker.values())
    overall = {
        "duration": span,
        "n_speakers": len(per_speaker),
        "total_words": total_words,
        "wpm": total_words / (voiced / 60.0) if voiced >= 5.0 else None,
        "interruptions": interruptions_total,
        "silence_ratio": silence_ratio,
        "silence_rating": _rate_silence(silence_ratio),
        "questions": sum(s["questions"] for s in per_speaker.values()),
    }

    feedback = _build_feedback(per_speaker, overall)
    return {"per_speaker": per_speaker, "overall": overall, "feedback": feedback}


def _build_feedback(per_speaker: dict, overall: dict) -> list[str]:
    """Evaluative Spanish feedback bullets.

    Every session gets context (✅/🟡/⚠️ against the orientative coaching
    ranges above), not just breach warnings — bare numbers don't tell the
    user whether they're good or bad. Sentiment is deliberately NOT repeated
    here; it has its own section in the UI/report.
    """
    feedback: list[str] = []
    minutes = overall["duration"] / 60.0
    n = overall["n_speakers"]

    def icon(rating: dict | None) -> str:
        return RATING_ICONS.get(rating["level"], "ℹ️") if rating else "ℹ️"

    if n == 1:
        _speaker, s = next(iter(per_speaker.items()))
        feedback.append(
            "🎙️ Un solo hablante detectado — métricas en modo exposición/presentación "
            "(el reparto e interrupciones no aplican)."
        )
        pr = s.get("pace_rating")
        if pr:
            feedback.append(f"{icon(pr)} {pr['note'][0].upper() + pr['note'][1:]}.")
        fr = s.get("fillers_rating")
        if fr and s["words"] >= 80:
            listed = ""
            if fr["level"] == "warn":
                top_fillers = sorted(s["fillers"].items(), key=lambda kv: -kv[1])[:2]
                listed = " — sobre todo " + ", ".join(f"«{w}» ({c}×)" for w, c in top_fillers)
            feedback.append(f"{icon(fr)} {fr['note'][0].upper() + fr['note'][1:]}{listed}.")
    else:
        top_speaker, top = max(per_speaker.items(), key=lambda kv: kv[1]["talk_share"])
        share = top["talk_share"]
        if share >= 0.65:
            feedback.append(
                f"⚠️ Speaker {top_speaker} dominó la conversación ({share * 100:.0f}% del habla; "
                "en una reunión equilibrada nadie supera ~55–60%)."
            )
        elif share <= 0.55:
            feedback.append(
                f"✅ Reparto equilibrado entre {n} participantes "
                f"(el que más habla, Speaker {top_speaker}, ocupa {share * 100:.0f}%)."
            )
        else:
            feedback.append(
                f"🟡 Reparto algo cargado hacia Speaker {top_speaker} "
                f"({share * 100:.0f}%; equilibrado es ≲55–60%)."
            )

        for speaker, s in sorted(per_speaker.items(), key=lambda kv: str(kv[0])):
            pr = s.get("pace_rating")
            if pr and pr["level"] != "good" and s["talk_time"] >= 30:
                feedback.append(f"{icon(pr)} Speaker {speaker}: {pr['note']}.")

        for speaker, s in sorted(per_speaker.items(), key=lambda kv: str(kv[0])):
            fr = s.get("fillers_rating")
            if fr and fr["level"] == "warn" and s["fillers_total"] >= 4:
                top_fillers = sorted(s["fillers"].items(), key=lambda kv: -kv[1])[:2]
                listed = ", ".join(f"«{w}» ({c}×)" for w, c in top_fillers)
                feedback.append(f"{icon(fr)} Speaker {speaker}: {fr['note']} — {listed}.")

        if minutes >= 2:
            rate = overall["interruptions"] / minutes if minutes > 0 else 0.0
            if overall["interruptions"] == 0:
                feedback.append("✅ Sin interrupciones: los turnos se respetaron.")
            elif rate > 1.0:
                feedback.append(
                    f"⚠️ {overall['interruptions']} interrupciones/solapamientos "
                    f"({rate:.1f}/min; fluido es <1/min)."
                )
            else:
                feedback.append(
                    f"🟡 {overall['interruptions']} interrupciones/solapamientos "
                    f"({rate:.1f}/min; fluido es <1/min)."
                )

        for speaker, s in sorted(per_speaker.items(), key=lambda kv: str(kv[0])):
            if s["longest_monologue"] >= 120:
                feedback.append(
                    f"⚠️ Monólogo de {_fmt_mmss(s['longest_monologue'])} de Speaker {speaker} "
                    "(turnos de <2 min mantienen mejor la atención)."
                )

    sr = overall.get("silence_rating")
    if sr:
        feedback.append(f"{icon(sr)} {sr['note'][0].upper() + sr['note'][1:]}.")

    return feedback


# ---------------------------------------------------------------------------
# High-level orchestration (single call OR chunked with speaker remapping)
# ---------------------------------------------------------------------------

def transcribe_file_deepgram(
    path: str,
    api_key: str,
    language: str | None = None,
    diarize: bool = False,
    want_sentiment: bool = False,
    want_insights: bool = False,
    max_chunk_mb: int = DEEPGRAM_MAX_CHUNK_MB,
    timeout: float = DEFAULT_TIMEOUT,
    progress_cb: ProgressCb | None = None,
) -> dict:
    """Transcribe a file of any size with Deepgram and optional analysis.

    Returns a result dict:
        transcript: formatted text (diarized `[Speaker N]:` lines or plain)
        utterances: normalized utterances (absolute times, global speakers)
        sentiment: summarize_sentiment() output or None
        insights: compute_speech_insights() output or None
        detected_language / model_used / duration: response metadata
        warnings: non-fatal issues worth surfacing to the user
    """

    def _progress(fraction: float | None, message: str) -> None:
        if progress_cb:
            progress_cb(fraction, message)

    warnings: list[str] = []
    features: dict[str, str] = {}
    need_utterances = diarize or want_insights or want_sentiment
    if need_utterances:
        # utterances/diarize give us word + speaker timing even without diarize UI.
        features["utterances"] = "true"
    if want_sentiment:
        features["sentiment"] = "true"
        if language == "en":
            features["filler_words"] = "true"
        if language not in (None, "en"):
            warnings.append(
                "El análisis de sentimiento de Deepgram solo está disponible en inglés; "
                "se omitió para este audio."
            )
    elif want_insights and language == "en":
        features["filler_words"] = "true"

    _progress(None, "🔍 Analyzing file...")
    chunk_paths, chunk_durations, temp_dir = _split_audio_with_durations(path, max_chunk_mb)

    try:
        if len(chunk_paths) == 1:
            _progress(0.2, "🎤 Transcribing with Deepgram...")
            raw = transcribe_with_deepgram(
                path, api_key, language=language, diarize=diarize,
                return_raw=True, features=features, timeout=timeout,
            )
            if isinstance(raw, dict) and "error" in raw:
                raise DeepgramError(str(raw["error"]))

            utterances = _normalize_utterances(get_utterances(raw))
            transcript = format_diarized_output(raw) if diarize else get_transcript(raw)
            sentiment_points = extract_sentiment_points(raw) if want_sentiment else []
            duration = get_audio_duration(raw)
            detected_language = get_detected_language(raw)
            model_used = get_model_used(raw)
        else:
            _progress(None, f"📦 File split into {len(chunk_paths)} chunks")
            merged = _transcribe_chunks(
                chunk_paths, chunk_durations, api_key, language, diarize,
                features if want_sentiment or need_utterances else {},
                want_sentiment, timeout, _progress,
            )
            transcript = merged["transcript"]
            utterances = merged["utterances"]
            sentiment_points = merged["sentiment_points"]
            duration = merged["duration"]
            detected_language = merged["detected_language"]
            model_used = merged["model_used"]
            warnings.extend(merged["warnings"])
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    sentiment = summarize_sentiment(sentiment_points) if sentiment_points else None
    if want_sentiment and sentiment is None and language in (None, "en"):
        warnings.append(
            "Deepgram no devolvió datos de sentimiento para este audio "
            "(la función solo existe para audio en inglés y modelos nova"
            + (f"; idioma detectado: {detected_language}" if detected_language else "")
            + ")."
        )

    insights = None
    if want_insights:
        effective_language = language or (
            "en" if (detected_language or "").startswith("en") else "es"
        )
        insights = compute_speech_insights(
            utterances, language=effective_language, total_duration=duration,
        )
        if insights is None:
            warnings.append(
                "No se pudieron calcular métricas de conversación (sin utterances "
                "en la respuesta)."
            )

    return {
        "transcript": transcript,
        "utterances": utterances,
        "sentiment": sentiment,
        "insights": insights,
        "detected_language": detected_language,
        "model_used": model_used,
        "duration": duration,
        "warnings": warnings,
    }


def _transcribe_chunks(
    chunk_paths: list[str],
    chunk_durations: list[float],
    api_key: str,
    language: str | None,
    diarize: bool,
    features: dict[str, str],
    want_sentiment: bool,
    timeout: float,
    progress: ProgressCb,
) -> dict:
    """Chunked transcription keeping speaker IDs consistent across chunks."""
    transcriptions: list[str] = []
    all_utterances: list[dict] = []
    sentiment_points: list[dict] = []
    warnings: list[str] = []
    detected_language = None
    model_used = None
    total_duration = 0.0

    prev_local_to_global: dict[int, int] = {}
    prev_stats: dict = {}
    prev_last_speaker: int | None = None
    next_global_speaker_id = 0
    offset_s = 0.0

    n = len(chunk_paths)
    for chunk_idx, chunk_path in enumerate(chunk_paths):
        chunk_num = chunk_idx + 1
        progress(chunk_num / n, f"🎤 Processing chunk {chunk_num}/{n}...")

        fallback_duration = (
            chunk_durations[chunk_idx] if chunk_idx < len(chunk_durations) else 0.0
        )
        try:
            raw = transcribe_with_deepgram(
                chunk_path, api_key, language=language, diarize=diarize,
                return_raw=True, features=features, timeout=timeout,
            )
        except DeepgramError as exc:
            raw = {"error": str(exc)}

        if isinstance(raw, dict) and "error" in raw:
            transcriptions.append(f"[Error in chunk {chunk_num}: {raw['error']}]")
            warnings.append(f"Chunk {chunk_num} falló: {raw['error']}")
            offset_s += fallback_duration
            prev_stats, prev_local_to_global, prev_last_speaker = {}, {}, None
            continue

        detected_language = detected_language or get_detected_language(raw)
        model_used = model_used or get_model_used(raw)
        chunk_duration = get_audio_duration(raw) or fallback_duration
        total_duration += chunk_duration

        if not diarize:
            transcriptions.append(get_transcript(raw))
            all_utterances.extend(_normalize_utterances(get_utterances(raw), offset_s))
            if want_sentiment:
                sentiment_points.extend(extract_sentiment_points(raw, offset_s))
            offset_s += chunk_duration
            continue

        utterances, current_stats = extract_speakers_from_response(raw)
        if not utterances:
            transcriptions.append(get_transcript(raw))
            offset_s += chunk_duration
            prev_stats, prev_local_to_global, prev_last_speaker = {}, {}, None
            continue

        current_first_speaker = utterances[0].get("speaker", 0)
        if chunk_idx > 0 and prev_stats:
            local_mapping = map_speakers_between_chunks(
                prev_stats, current_stats, prev_last_speaker, current_first_speaker
            )
        else:
            local_mapping = {}

        # Translate local IDs to stable global IDs.
        speaker_mapping: dict[int, int] = {}
        for local_speaker in sorted(current_stats.keys()):
            mapped_prev = local_mapping.get(local_speaker)
            if mapped_prev is not None and mapped_prev in prev_local_to_global:
                speaker_mapping[local_speaker] = prev_local_to_global[mapped_prev]
            else:
                speaker_mapping[local_speaker] = next_global_speaker_id
                next_global_speaker_id += 1

        transcriptions.append(format_diarized_output(raw, speaker_mapping))
        all_utterances.extend(_normalize_utterances(utterances, offset_s, speaker_mapping))
        if want_sentiment:
            sentiment_points.extend(extract_sentiment_points(raw, offset_s, speaker_mapping))

        prev_local_to_global = speaker_mapping
        prev_stats = current_stats
        prev_last_speaker = utterances[-1].get("speaker", 0)
        offset_s += chunk_duration
        progress(chunk_num / n, f"✅ Chunk {chunk_num} processed")

    return {
        "transcript": "\n\n".join(transcriptions) if diarize else " ".join(transcriptions),
        "utterances": all_utterances,
        "sentiment_points": sentiment_points,
        "duration": total_duration or None,
        "detected_language": detected_language,
        "model_used": model_used,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Text report (shared by the app download button and the CLI --report)
# ---------------------------------------------------------------------------

def build_report(result: dict, filename: str = "") -> str:
    """Render a full Spanish text report: transcript + insights + sentiment."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"INFORME DE TRANSCRIPCIÓN{' — ' + filename if filename else ''}")
    meta_bits = []
    if result.get("model_used"):
        meta_bits.append(f"modelo: {result['model_used']}")
    if result.get("detected_language"):
        meta_bits.append(f"idioma detectado: {result['detected_language']}")
    if result.get("duration"):
        meta_bits.append(f"duración: {_fmt_mmss(result['duration'])}")
    if meta_bits:
        lines.append(" · ".join(meta_bits))
    lines.append("=" * 60)

    insights = result.get("insights")
    if insights:
        lines.append("")
        lines.append("── ANÁLISIS DE CONVERSACIÓN ──")
        for speaker, s in sorted(insights["per_speaker"].items(), key=lambda kv: str(kv[0])):
            wpm = f"{s['wpm']:.0f} ppm" if s["wpm"] else "—"
            fillers = (
                ", ".join(f"«{w}»×{c}" for w, c in sorted(s["fillers"].items(), key=lambda kv: -kv[1])[:3])
                or "—"
            )
            lines.append(
                f"Speaker {speaker}: {s['talk_share'] * 100:.0f}% del habla "
                f"({_fmt_mmss(s['talk_time'])}), {s['words']} palabras, ritmo {wpm}, "
                f"{s['turns']} turnos, {s['questions']} preguntas, muletillas: {fillers}"
            )
            ratings = [s.get("pace_rating"), s.get("fillers_rating")]
            notes = "; ".join(
                f"{RATING_ICONS.get(r['level'], 'ℹ️')} {r['note']}" for r in ratings if r
            )
            if notes:
                lines.append(f"    {notes}")
        overall = insights["overall"]
        lines.append(
            f"Global: {overall['n_speakers']} hablantes · {overall['total_words']} palabras · "
            f"{overall['interruptions']} interrupciones · "
            f"{overall['silence_ratio'] * 100:.0f}% silencio"
        )
        lines.append("")
        lines.append("Observaciones:")
        for item in insights["feedback"]:
            lines.append(f"  • {item}")

    sentiment = result.get("sentiment")
    if sentiment:
        lines.append("")
        lines.append("── SENTIMIENTO (Deepgram) ──")
        avg = sentiment["average"]
        label_es = {"positive": "positivo", "neutral": "neutral", "negative": "negativo"}.get(
            avg["sentiment"], avg["sentiment"]
        )
        lines.append(
            f"Tono general: {label_es} (score {avg['sentiment_score']:+.2f} — "
            f"{describe_sentiment_score(avg['sentiment_score'])})"
        )
        for speaker, sp in sorted(sentiment["per_speaker"].items(), key=lambda kv: str(kv[0])):
            label_sp = {"positive": "positivo", "neutral": "neutral", "negative": "negativo"}.get(
                sp["sentiment"], sp["sentiment"]
            )
            lines.append(
                f"  Speaker {speaker}: {label_sp} ({sp['sentiment_score']:+.2f} — "
                f"{describe_sentiment_score(sp['sentiment_score'])})"
            )
        emoji_line = sentiment_timeline_emoji(sentiment)
        if emoji_line:
            lines.append(f"  Evolución: {emoji_line}")

    if insights or sentiment:
        lines.append("")
        lines.append("── CÓMO LEER ESTAS MÉTRICAS ──")
        for guide_line in METRIC_GUIDE_ES:
            lines.append(f"  • {guide_line}")

    for warning in result.get("warnings", []):
        lines.append("")
        lines.append(f"⚠️ {warning}")

    lines.append("")
    lines.append("── TRANSCRIPCIÓN ──")
    lines.append(result.get("transcript", ""))
    return "\n".join(lines)


def sentiment_timeline_emoji(sentiment: dict) -> str:
    """Compact emoji sparkline of the sentiment timeline (mobile-friendly)."""
    timeline = sentiment.get("timeline") or []
    out = []
    for bucket in timeline:
        score = bucket.get("score")
        if score is None:
            out.append("·")
        elif score >= 0.33:
            out.append("😊")
        elif score <= -0.33:
            out.append("🙁")
        else:
            out.append("😐")
    return "".join(out)
