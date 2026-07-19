#!/usr/bin/env python
"""
CLI: Deepgram transcription with optional diarization, sentiment and insights.

Converts input audio/video to MP3, transcribes with Deepgram (nova models with
automatic fallback), and — when diarization is on — formats output as
speaker-labeled lines with speaker IDs kept consistent across chunks.

All transcription/diarization/analysis logic lives in core/transcription.py
(shared with the Streamlit app — ROADMAP Phase 0).

Examples:
  python scripts/deepgram_transcribe_cli.py --file "C:\\path\\video.mp4"
  python scripts/deepgram_transcribe_cli.py --file "C:\\path\\audio.wav" --language es
  python scripts/deepgram_transcribe_cli.py --file "C:\\path\\audio.mp3" --no-diarize
  python scripts/deepgram_transcribe_cli.py --file meeting.mp3 --language en --sentiment

Contract: stdout is the transcript; stderr is progress/metadata. Output files:
  output/<stem>_transcript.txt          (always)
  output/<stem>_report.txt              (with --insights/--sentiment)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
from dotenv import load_dotenv
from pydub import AudioSegment

# Make the repo root importable when invoked as `python scripts/...py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import transcription as tr  # noqa: E402

load_dotenv()

# Patch pydub to use bundled ffmpeg/ffprobe (works better on Windows than PATH-only).
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = ffmpeg_path
AudioSegment.ffmpeg = ffmpeg_path


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
        description="Transcribe audio/video using Deepgram, with MP3 conversion, "
        "diarization, sentiment analysis and conversation insights.",
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
        "--sentiment",
        dest="sentiment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request Deepgram sentiment analysis (English audio only).",
    )
    parser.add_argument(
        "--insights",
        dest="insights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute conversation insights (talk share, pace, interruptions, fillers).",
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=tr.DEEPGRAM_MAX_CHUNK_MB,
        help="Approximate max size before chunking into smaller MP3 files. "
        "Deepgram accepts large files, so chunking (which is what makes "
        "speaker IDs need remapping) is rare by default.",
    )

    args = parser.parse_args()

    input_file = Path(args.file)
    if not input_file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        print("Error: DEEPGRAM_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_file.stem}_transcript.txt"

    def progress(_fraction: float | None, message: str) -> None:
        print(message, file=sys.stderr)

    mp3_path, cleanup_paths = ensure_mp3(str(input_file))
    try:
        print(f"Using MP3 for transcription: {mp3_path}", file=sys.stderr)
        result = tr.transcribe_file_deepgram(
            mp3_path,
            api_key,
            language=args.language,
            diarize=args.diarize,
            want_sentiment=args.sentiment,
            want_insights=args.insights,
            max_chunk_mb=args.max_size_mb,
            progress_cb=progress,
        )

        transcript_str = result["transcript"]
        output_path.write_text(transcript_str, encoding="utf-8")

        if result.get("model_used"):
            print(f"Model used: {result['model_used']}", file=sys.stderr)
        if result.get("detected_language"):
            print(f"Detected language: {result['detected_language']}", file=sys.stderr)
        for warning in result.get("warnings", []):
            print(f"Warning: {warning}", file=sys.stderr)

        if result.get("insights") or result.get("sentiment"):
            report_path = output_dir / f"{input_file.stem}_report.txt"
            report_path.write_text(
                tr.build_report(result, filename=input_file.name), encoding="utf-8"
            )
            print(f"Saved report to: {report_path}", file=sys.stderr)
            for item in (result.get("insights") or {}).get("feedback", []):
                print(f"  • {item}", file=sys.stderr)

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
