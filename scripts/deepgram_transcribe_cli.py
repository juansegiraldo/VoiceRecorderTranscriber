#!/usr/bin/env python
"""
CLI: Deepgram transcription with optional diarization.

Converts input audio/video to MP3, transcribes with Deepgram, and (when diarize=true)
formats output as speaker-labeled lines while keeping speaker IDs consistent across chunks.

Examples:
  python scripts/deepgram_transcribe_cli.py --file "C:\\path\\video.mp4"
  python scripts/deepgram_transcribe_cli.py --file "C:\\path\\audio.wav" --language es
  python scripts/deepgram_transcribe_cli.py --file "C:\\path\\audio.mp3" --no-diarize
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import requests
from dotenv import load_dotenv
from pydub import AudioSegment


load_dotenv()

# Patch pydub to use bundled ffmpeg/ffprobe (works better on Windows than PATH-only).
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = ffmpeg_path
AudioSegment.ffmpeg = ffmpeg_path


def split_audio_file(file_path: str, max_size_mb: int = 24) -> list[str]:
    """
    Split a large audio file into smaller MP3 chunks.

    Note: chunk size is approximated by splitting by duration based on file size; it is
    enough for fitting common API limits.
    """

    max_size_bytes = max_size_mb * 1024 * 1024
    file_size = os.path.getsize(file_path)
    if file_size <= max_size_bytes:
        return [file_path]

    audio = AudioSegment.from_file(file_path)
    num_chunks = max(2, int((file_size / max_size_bytes) + 0.999999))
    chunk_duration = len(audio) // num_chunks

    temp_dir = tempfile.mkdtemp(prefix="deepgram_chunks_")
    chunk_paths: list[str] = []
    try:
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = start_time + chunk_duration if i < num_chunks - 1 else len(audio)
            chunk = audio[start_time:end_time]
            chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.mp3")
            chunk.export(chunk_path, format="mp3")
            chunk_paths.append(chunk_path)
        return chunk_paths
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def format_diarized_output(
    deepgram_response: dict[str, Any], speaker_mapping: dict[int, int] | None = None
) -> str:
    """
    Turn Deepgram diarized utterances into:
      [Speaker X]: text
    with consecutive utterances grouped.
    """

    utterances = deepgram_response.get("results", {}).get("utterances", [])
    if not utterances:
        transcript = (
            deepgram_response.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        )
        return transcript if transcript else ""

    formatted_lines: list[str] = []
    current_speaker: int | None = None
    current_text_parts: list[str] = []

    for utterance in utterances:
        speaker = int(utterance.get("speaker", 0))
        if speaker_mapping and speaker in speaker_mapping:
            speaker = int(speaker_mapping[speaker])

        transcript = utterance.get("transcript", "").strip()
        if not transcript:
            continue

        if speaker == current_speaker:
            current_text_parts.append(transcript)
        else:
            if current_speaker is not None and current_text_parts:
                combined_text = " ".join(current_text_parts)
                formatted_lines.append(f"[Speaker {current_speaker}]: {combined_text}")

            current_speaker = speaker
            current_text_parts = [transcript]

    if current_speaker is not None and current_text_parts:
        combined_text = " ".join(current_text_parts)
        formatted_lines.append(f"[Speaker {current_speaker}]: {combined_text}")

    return "\n".join(formatted_lines)


def extract_speakers_from_response(
    deepgram_response: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    utterances = deepgram_response.get("results", {}).get("utterances", [])
    if not utterances:
        return [], {}

    speaker_stats: dict[int, int] = {}
    for utterance in utterances:
        speaker = int(utterance.get("speaker", 0))
        speaker_stats[speaker] = speaker_stats.get(speaker, 0) + 1

    return utterances, speaker_stats


def map_speakers_between_chunks(
    prev_speakers: dict[int, int],
    current_speakers: dict[int, int],
    prev_last_speaker: int | None = None,
) -> dict[int, int]:
    """
    Map current chunk local speaker IDs to previous chunk IDs to keep consistency.
    """

    if not prev_speakers or not current_speakers:
        return {}

    mapping: dict[int, int] = {}

    if prev_last_speaker is not None:
        most_common_current = max(current_speakers.items(), key=lambda x: x[1])[0]
        if prev_last_speaker in prev_speakers:
            mapping[most_common_current] = prev_last_speaker

    prev_sorted = sorted(prev_speakers.items(), key=lambda x: x[1], reverse=True)
    current_sorted = sorted(current_speakers.items(), key=lambda x: x[1], reverse=True)

    for i, (current_speaker, _) in enumerate(current_sorted):
        if current_speaker in mapping:
            continue
        if i < len(prev_sorted):
            mapping[current_speaker] = prev_sorted[i][0]
        else:
            max_prev_speaker = max(prev_speakers.keys()) if prev_speakers else -1
            mapping[current_speaker] = max_prev_speaker + 1

    return mapping


def transcribe_with_deepgram(
    path: str,
    language: str | None = None,
    diarize: bool = False,
    return_raw: bool = False,
) -> str | dict[str, Any]:
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPGRAM_API_KEY environment variable not set")

    base_params = "smart_format=true&punctuate=true"
    if diarize:
        base_params += "&diarize=true&utterances=true"
    else:
        base_params += "&diarize=false"

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/mp3",
        "Accept": "application/json",
    }

    with open(path, "rb") as f:
        audio_data = f.read()

    def _do_request(url: str) -> dict[str, Any]:
        response = requests.post(url, headers=headers, data=audio_data, timeout=300)
        if not response.ok:
            if return_raw:
                return {"error": response.text}
            raise RuntimeError(f"Deepgram error: {response.text}")
        return response.json()

    # Attempt strategy (mirrors AppTranscribe.py idea):
    # 1) If language is specified -> try that language.
    # 2) Always have a detect_language fallback.
    # 3) Final fallback to the plain endpoint.
    lang_code = None
    if language in ("es", "en"):
        lang_code = language

    attempts: list[tuple[str, str]] = []
    model_param = "&model=base"

    if lang_code is not None:
        attempts.append(
            (
                "default",
                f"https://api.deepgram.com/v1/listen?{base_params}&language={lang_code}{model_param}",
            )
        )

    attempts.append(
        (
            "auto",
            f"https://api.deepgram.com/v1/listen?{base_params}&detect_language=true{model_param}",
        )
    )
    attempts.append(("original", f"https://api.deepgram.com/v1/listen?{base_params}"))

    last_error: str | None = None
    for attempt_name, url in attempts:
        dg = _do_request(url)
        if "error" in dg:
            last_error = str(dg["error"])
            continue

        if return_raw:
            return dg

        if diarize:
            formatted = format_diarized_output(dg)
            if formatted and len(formatted.strip()) > 10:
                return formatted
        else:
            transcript = (
                dg.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            )
            if transcript and len(transcript.strip()) > 10:
                return transcript

    if return_raw:
        return {"error": last_error or "Deepgram failed to transcribe properly"}

    raise RuntimeError(
        "Deepgram failed to transcribe properly after multiple attempts"
        + (f": {last_error}" if last_error else "")
    )


def transcribe_large_file_with_diarization(
    chunk_paths: list[str],
    language: str | None = None,
) -> str:
    transcriptions: list[str] = []
    temp_dir = os.path.dirname(chunk_paths[0]) if len(chunk_paths) > 1 else None

    global_speaker_mapping: dict[tuple[int, int], int] = {}
    prev_speakers: dict[int, int] = {}
    prev_last_speaker: int | None = None
    next_global_speaker_id = 0

    try:
        for chunk_idx, chunk_path in enumerate(chunk_paths):
            chunk_num = chunk_idx + 1
            print(f"Deepgram diarization: chunk {chunk_num}/{len(chunk_paths)}", file=sys.stderr)

            raw_response = transcribe_with_deepgram(
                chunk_path, language=language, diarize=True, return_raw=True
            )
            if isinstance(raw_response, dict) and "error" in raw_response:
                transcriptions.append(f"[Error in chunk {chunk_num}: {raw_response['error']}]")
                continue

            utterances, current_speakers = extract_speakers_from_response(raw_response)
            if not utterances:
                chunk_transcription = transcribe_with_deepgram(
                    chunk_path, language=language, diarize=False, return_raw=False
                )
                transcriptions.append(str(chunk_transcription))
                continue

            speaker_mapping: dict[int, int] = {}

            if chunk_idx > 0 and prev_speakers:
                local_mapping = map_speakers_between_chunks(
                    prev_speakers, current_speakers, prev_last_speaker
                )

                for local_speaker, mapped_speaker in local_mapping.items():
                    prev_global_id: int | None = None
                    for (prev_chunk_idx, prev_local_speaker), global_id in global_speaker_mapping.items():
                        if prev_chunk_idx == chunk_idx - 1 and prev_local_speaker == mapped_speaker:
                            prev_global_id = global_id
                            break

                    if prev_global_id is not None:
                        speaker_mapping[local_speaker] = prev_global_id
                    else:
                        speaker_mapping[local_speaker] = next_global_speaker_id
                        next_global_speaker_id += 1
            else:
                for local_speaker in current_speakers.keys():
                    speaker_mapping[local_speaker] = next_global_speaker_id
                    next_global_speaker_id += 1

            for local_speaker, global_speaker in speaker_mapping.items():
                global_speaker_mapping[(chunk_idx, local_speaker)] = global_speaker

            chunk_transcription = format_diarized_output(raw_response, speaker_mapping)
            transcriptions.append(chunk_transcription)

            prev_speakers = current_speakers
            if utterances:
                prev_last_speaker = int(utterances[-1].get("speaker", 0))
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    return "\n\n".join(transcriptions)


def transcribe_large_file(
    file_path: str,
    language: str | None = None,
    diarize: bool = False,
    max_size_mb: int = 24,
) -> str:
    chunk_paths = split_audio_file(file_path, max_size_mb=max_size_mb)
    if len(chunk_paths) == 1:
        return str(
            transcribe_with_deepgram(file_path, language=language, diarize=diarize, return_raw=False)
        )

    if diarize:
        return transcribe_large_file_with_diarization(chunk_paths, language=language)

    transcriptions: list[str] = []
    temp_dir = os.path.dirname(chunk_paths[0]) if len(chunk_paths) > 1 else None
    try:
        for i, chunk_path in enumerate(chunk_paths, 1):
            print(f"Deepgram transcription: chunk {i}/{len(chunk_paths)}", file=sys.stderr)
            chunk_transcription = transcribe_with_deepgram(
                chunk_path, language=language, diarize=False, return_raw=False
            )
            transcriptions.append(str(chunk_transcription))
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    return " ".join(transcriptions)


def ensure_mp3(input_path: str) -> tuple[str, list[str]]:
    """
    Convert input audio/video to MP3 if needed.

    Returns: (mp3_path, cleanup_paths)
    where cleanup_paths are temp files that should be deleted by the caller.
    """

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(str(input_path))

    suffix = input_file.suffix.lower()
    if suffix == ".mp3":
        return str(input_file), []

    cleanup_paths: list[str] = []
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    out_tmp.close()
    out_path = out_tmp.name
    cleanup_paths.append(out_path)

    if suffix in (".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"):
        # pydub can decode many formats once ffmpeg is available.
        audio = AudioSegment.from_file(str(input_file))
        audio.export(out_path, format="mp3", bitrate="192k")
        return out_path, cleanup_paths

    if suffix == ".mp4":
        # Use ffmpeg (bundled) for extraction.
        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            str(input_file),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ab",
            "192k",
            "-ar",
            "44100",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_path, cleanup_paths

    raise ValueError(f"Unsupported input extension: {suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video using Deepgram, with MP3 conversion and diarization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--file", required=True, help="Path to the input audio/video file.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where <stem>_transcript.txt will be written.",
    )
    parser.add_argument("--language", choices=["es", "en"], default=None, help="Language hint.")
    parser.add_argument(
        "--diarize",
        dest="diarize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable speaker diarization (Deepgram utterances).",
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=24,
        help="Approximate max size before chunking into smaller MP3 files.",
    )

    args = parser.parse_args()

    input_file = Path(args.file)
    if not input_file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_file.stem}_transcript.txt"

    mp3_path, cleanup_paths = ensure_mp3(str(input_file))
    try:
        print(f"Using MP3 for transcription: {mp3_path}", file=sys.stderr)
        transcript = transcribe_large_file(
            mp3_path,
            language=args.language,
            diarize=args.diarize,
            max_size_mb=args.max_size_mb,
        )
        transcript_str = str(transcript)

        output_path.write_text(transcript_str, encoding="utf-8")
        # Contract: stdout is the transcript; stderr is metadata/logging.
        print(transcript_str)
        print(f"Saved transcript to: {output_path}", file=sys.stderr)
    finally:
        for p in cleanup_paths:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


if __name__ == "__main__":
    main()

