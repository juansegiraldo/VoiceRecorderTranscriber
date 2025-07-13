# Voice Transcriber

A powerful audio transcription application built with Streamlit that supports multiple audio formats and transcription services.

## 🚀 Features

- **Multiple Audio Formats**: MP3, WAV, M4A (with automatic conversion)
- **Transcription Services**: OpenAI Whisper and Deepgram
- **Web Interface**: Beautiful Streamlit-based UI
- **Automatic Conversion**: M4A files are automatically converted to MP3 before transcription
- **Large File Support**: Automatic chunking for files larger than 24MB
- **Progress Tracking**: Real-time progress updates during transcription

## 📁 Project Structure

```
VoiceTranscriber/
├── AppTranscribe.py          # Main Streamlit application
├── scripts/                  # Supporting scripts and utilities
│   ├── convert_m4a_to_mp3.py    # M4A to MP3 converter
│   ├── transcribe.py             # Command-line transcription script
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
- Python 3.8 or higher
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

#### Command Line Transcription
```bash
# Process all files in input/ folder
python scripts/transcribe.py

# Process specific file
python scripts/transcribe.py --file "path/to/audio.wav"
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
   - Better for real-time transcription
   - Supports multiple languages
   - Good for Spanish content

2. **OpenAI Whisper**
   - Excellent accuracy
   - Great for non-English content
   - Requires OpenAI API key

### File Size Limits

- **Default**: 24MB maximum file size
- **Large files**: Automatically split into chunks
- **Configurable**: Adjustable in the web interface

## 📖 Examples

### Web Application
1. Open the app in your browser
2. Select your preferred transcription model
3. Upload an audio file (MP3, WAV, or M4A)
4. Click "Start Transcription"
5. Download the results

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

