#!/usr/bin/env python3
"""
MP4 to MP3 Converter using ffmpeg (robust for large/complex files)

Usage:
    python convert_mp4_to_mp3_ffmpeg.py <input_path> [output_path]

Examples:
    python convert_mp4_to_mp3_ffmpeg.py video.mp4
    python convert_mp4_to_mp3_ffmpeg.py video.mp4 audio.mp3
    python convert_mp4_to_mp3_ffmpeg.py input_folder/ output_folder/

If no arguments are given, processes all MP4s in 'input/' to 'output/'.
"""
import os
import sys
import subprocess
from pathlib import Path

def convert_mp4_to_mp3_ffmpeg(input_path, output_path=None, bitrate="192k"):
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Input file not found: {input_path}")
        return False
    if output_path is None:
        output_file = input_file.with_suffix('.mp3')
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting {input_file} -> {output_file}")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_file), "-vn",
        "-acodec", "libmp3lame", "-ab", bitrate, str(output_file)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Done: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.decode(errors='ignore')}")
        return False

def convert_directory(input_dir, output_dir=None, bitrate="192k"):
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Input directory not found: {input_dir}")
        return 0, 0
    mp4_files = list(input_path.glob("*.mp4"))
    if not mp4_files:
        print(f"No MP4 files found in {input_dir}")
        return 0, 0
    print(f"Found {len(mp4_files)} MP4 files to convert.")
    success = 0
    for mp4_file in mp4_files:
        if output_dir is None:
            output_file = mp4_file.with_suffix('.mp3')
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"{mp4_file.stem}.mp3"
        if convert_mp4_to_mp3_ffmpeg(mp4_file, output_file, bitrate):
            success += 1
    print(f"Converted {success}/{len(mp4_files)} files.")
    return success, len(mp4_files)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert MP4 to MP3 using ffmpeg.")
    parser.add_argument("input_path", nargs="?", help="Input MP4 file or directory")
    parser.add_argument("output_path", nargs="?", help="Output MP3 file or directory")
    parser.add_argument("--bitrate", default="192k", help="MP3 bitrate (default: 192k)")
    args = parser.parse_args()
    if not args.input_path:
        # Interactive mode: input/ to output/
        input_dir = Path("input")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        convert_directory(input_dir, output_dir, args.bitrate)
        return
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Input path does not exist: {args.input_path}")
        sys.exit(1)
    if input_path.is_file():
        convert_mp4_to_mp3_ffmpeg(input_path, args.output_path, args.bitrate)
    elif input_path.is_dir():
        convert_directory(input_path, args.output_path, args.bitrate)
    else:
        print(f"Input path is neither a file nor directory: {args.input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main() 