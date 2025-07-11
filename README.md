# Voice Recorder Transcriber

This project provides a simple command-line tool to send an audio file to OpenAI's Whisper API and print the transcription.

## Requirements
- Python 3.8+
- An OpenAI API key available in the environment variable `OPENAI_API_KEY`
- The `openai` Python package (`pip install openai`)

## Usage
```
python transcribe.py path/to/audio_file
```

The script uploads the specified audio file to OpenAI and outputs the transcribed text.

