# Voice Transcriber

A powerful audio transcription application built with Streamlit that supports multiple audio formats and transcription services.

## 🚀 Features

- **Multiple Audio Formats**: MP3, WAV, M4A, MP4 (with automatic conversion)
- **Transcription Services**: Deepgram (nova-3 with automatic fallback) and OpenAI Whisper
- **Speaker Diarization**: `[Speaker N]:` labels via Deepgram's batch diarizer v2, with speaker IDs kept consistent across chunks
- **Conversation Insights**: per-speaker talk share, pace (words/min), interruptions, monologues, filler words (muletillas) + feedback bullets
- **Sentiment Analysis**: Deepgram sentiment (English audio only) — overall, per speaker, and a timeline
- **Web Interface**: Mobile-first, Spanish-first Streamlit UI (v2 redesign: single Audio → Ajustes → Resultado flow, results in tabs — see [docs/rediseno-ux-v2.md](docs/rediseno-ux-v2.md)) with trimming and Google Drive input
- **Large File Support**: Deepgram accepts big files directly (chunking only >150MB, with silence-aware cuts); Whisper chunks at 24MB
- **Progress Tracking**: Real-time progress updates during transcription

## 📁 Project Structure

```
VoiceTranscriber/
├── AppTranscribe.py          # Main Streamlit application (UI + upload/trim flow)
├── core/
│   └── transcription.py      # Shared Deepgram + diarization + analysis logic
├── tests/
│   └── test_transcription.py # Unit tests (python -m unittest discover tests)
├── scripts/                  # Supporting scripts and utilities
│   ├── deepgram_transcribe_cli.py # Deepgram CLI (diarization/sentiment/insights)
│   ├── convert_m4a_to_mp3.py    # M4A to MP3 converter
│   ├── transcribe.py             # Legacy OpenAI-only batch script
│   ├── setup.ps1                 # Windows PowerShell setup
│   ├── setup.bat                 # Windows batch setup
│   └── setup.sh                  # Linux/Mac setup
├── m4a_input/                # M4A files for conversion
├── mp3_output/               # Converted MP3 files
├── input/                    # Audio files for transcription
├── output/                   # Transcription results
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🛠️ Setup

### Prerequisites
- Python 3.9 or higher (Streamlit >=1.40)
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/juansegiraldo/VoiceRecorderTranscriber.git
cd VoiceRecorderTranscriber
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up API keys**:
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

## 🎤 Usage

### Web Application (Recommended)

Run the main Streamlit application:
```bash
py -m streamlit run AppTranscribe.py
```

Then open your browser to `http://localhost:8501`

### Command Line Scripts

#### M4A to MP3 Converter
```bash
# Interactive mode (processes m4a_input/ to mp3_output/)
python scripts/convert_m4a_to_mp3.py

# Command line mode
python scripts/convert_m4a_to_mp3.py input.m4a output.mp3
```

#### Command Line Transcription (Deepgram, recommended)
```bash
# Diarized transcript + conversation report (output/<stem>_transcript.txt / _report.txt)
python scripts/deepgram_transcribe_cli.py --file "path/to/meeting.mp4" --language es

# English audio with Deepgram sentiment analysis
python scripts/deepgram_transcribe_cli.py --file meeting.mp3 --language en --sentiment

# Plain transcript, no diarization/insights
python scripts/deepgram_transcribe_cli.py --file audio.wav --no-diarize --no-insights
```

#### Command Line Transcription (legacy Whisper batch)
```bash
# Process all files in input/ folder
python scripts/transcribe.py

# Process specific file
python scripts/transcribe.py --file "path/to/audio.wav"
```

#### Tests
```bash
python -m unittest discover tests
```

## 📋 Supported Formats

### Web Application
- **MP3** (.mp3) - Direct transcription
- **WAV** (.wav) - Direct transcription
- **M4A** (.m4a) - Automatic conversion to MP3, then transcription

### Command Line Scripts
- **MP3** (.mp3)
- **WAV** (.wav)
- **M4A** (.m4a) - with conversion support
- **FLAC** (.flac)
- **OGG** (.ogg)
- **AAC** (.aac)
- **WMA** (.wma)

## 🔧 Configuration

### Transcription Models

1. **Deepgram** (Default)
   - nova-3 first, with automatic fallback (nova-2 → base → language auto-detect)
   - Speaker diarization (batch diarizer v2), conversation insights, sentiment (English)
   - Good for Spanish content (`language=es` on nova-3)

2. **OpenAI Whisper**
   - Excellent accuracy
   - Great for non-English content
   - Requires OpenAI API key

### File Size Limits

- **Deepgram**: files are sent in a single request up to ~150MB (Deepgram accepts up to 2GB); larger files are split at silence points and speaker IDs are remapped across chunks
- **OpenAI Whisper**: automatically split into chunks above 24MB (25MB API limit)

## 📖 Examples

### Web Application
1. Open the app in your browser
2. Upload an audio file (or load one from Google Drive)
3. Optionally adjust the language, trim range, or the model/analysis toggles under "Opciones avanzadas"
4. Click "Transcribir"
5. Read the result in the Transcripción / Métricas / Sentimiento tabs and download the .txt or the full report

### M4A Conversion
1. Place M4A files in `m4a_input/` folder
2. Run: `python scripts/convert_m4a_to_mp3.py`
3. Find converted MP3 files in `mp3_output/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [OpenAI Whisper](https://openai.com/research/whisper) for transcription
- [Deepgram](https://deepgram.com/) for transcription services
- [pydub](https://github.com/jiaaro/pydub) for audio processing

