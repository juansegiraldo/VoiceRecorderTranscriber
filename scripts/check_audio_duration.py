#!/usr/bin/env python3
"""
Simple script to check audio/video file duration
"""

import sys
from pathlib import Path
from pydub import AudioSegment

def check_duration(file_path):
    """Check the duration of an audio/video file"""
    try:
        audio = AudioSegment.from_file(file_path)
        duration_seconds = len(audio) / 1000.0
        duration_minutes = duration_seconds / 60.0
        
        print(f"File: {Path(file_path).name}")
        print(f"Duration: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes)")
        print(f"File size: {Path(file_path).stat().st_size / (1024*1024):.2f} MB")
        print("-" * 50)
        return duration_seconds
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_audio_duration.py <file1> [file2] ...")
        sys.exit(1)
    
    durations = []
    for file_path in sys.argv[1:]:
        duration = check_duration(file_path)
        if duration:
            durations.append(duration)
    
    if len(durations) >= 2:
        print(f"Duration difference: {abs(durations[0] - durations[1]):.2f} seconds")
        if abs(durations[0] - durations[1]) < 1:
            print("✅ Durations match (within 1 second)")
        else:
            print("❌ Durations don't match") 