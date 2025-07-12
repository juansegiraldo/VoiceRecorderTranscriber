# 🎤 Voice Transcriber - Streamlit App

A user-friendly web application for transcribing audio files using OpenAI's Whisper API, built with Streamlit.

## ✨ Features

- **Easy-to-use web interface** - No command line required
- **Drag & drop file upload** - Simply upload your audio files
- **Real-time progress tracking** - See transcription progress in real-time
- **Large file support** - Automatically splits large files into chunks
- **Multiple format support** - MP3, WAV, M4A, FLAC, OGG, AAC, WMA
- **Download results** - Save transcriptions as text files
- **Copy to clipboard** - Easy sharing of transcription results

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone or download this repository**

2. **Run the setup script (Windows PowerShell):**
   ```powershell
   .\setup_streamlit.ps1
   ```

3. **Add your OpenAI API key:**
   - Open the `.env` file
   - Replace `your_api_key_here` with your actual OpenAI API key
   - Get your API key from: https://platform.openai.com/api-keys

4. **Run the app:**
   ```powershell
   py -m streamlit run AppTranscribe.py
   ```

5. **Open your browser:**
   - The app will automatically open in your default browser
   - Usually at: http://localhost:8501

## 📖 Usage

1. **Upload Audio File:**
   - Click "Browse files" or drag & drop your audio file
   - Supported formats: MP3, WAV, M4A, FLAC, OGG, AAC, WMA

2. **Start Transcription:**
   - Click the "🎤 Start Transcription" button
   - Watch the progress bar and status updates

3. **Get Results:**
   - View the transcription in the text area
   - Download as a text file using the download button
   - Copy to clipboard for easy sharing

## ⚙️ Configuration

### Sidebar Options

- **OpenAI API Key:** Enter your API key directly in the app
- **Max File Size:** Adjust the chunking threshold (default: 24MB)
- **Supported Formats:** See all supported audio formats

### Environment Variables

You can also set your API key in the `.env` file:
```
OPENAI_API_KEY=your_actual_api_key_here
```

## 🔧 Manual Installation

If you prefer to install manually:

1. **Install dependencies:**
   ```powershell
   py -m pip install -r requirements.txt
   ```

2. **Create .env file:**
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Run the app:**
   ```powershell
   py -m streamlit run AppTranscribe.py
   ```

## 📁 File Structure

```
VoiceTranscriber/
├── AppTranscribe.py          # Streamlit web app
├── transcribe.py             # Original command-line script
├── requirements.txt          # Python dependencies
├── setup_streamlit.ps1      # Windows setup script
├── README_Streamlit.md      # This file
├── input/                   # Input folder (for CLI version)
└── output/                  # Output folder (for CLI version)
```

## 🎯 Features in Detail

### Large File Handling
- Automatically detects files larger than 24MB
- Splits them into manageable chunks
- Processes each chunk separately
- Combines results seamlessly

### Progress Tracking
- Real-time progress bar
- Status messages for each step
- Clear success/error indicators

### User Experience
- Modern, responsive interface
- Drag & drop file upload
- Instant feedback
- Easy download and sharing options

## 🛠️ Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**
   - Add your API key in the sidebar or .env file
   - Make sure the key is valid and has credits

2. **"Error during transcription"**
   - Check your internet connection
   - Verify your API key is correct
   - Ensure the audio file is not corrupted

3. **App won't start**
   - Make sure all dependencies are installed
   - Check Python version (3.8+ required)
   - Try running: `py -m pip install --upgrade streamlit`

### Getting Help

- Check the console output for detailed error messages
- Ensure your OpenAI API key has sufficient credits
- Verify the audio file format is supported

## 📄 License

This project is open source and available under the same license as the original project.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Enjoy transcribing your audio files with ease! 🎤✨** 