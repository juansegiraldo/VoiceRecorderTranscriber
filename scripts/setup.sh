#!/bin/bash

# Voice Transcriber Setup Script
# This script sets up the environment for the voice transcription project

echo "🎤 Voice Transcriber Setup Script"
echo "=================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "✅ pip3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "📚 Installing required packages..."
pip install openai python-dotenv pydub

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "🔑 Creating .env file template..."
    cat > .env << EOF
# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here
EOF
    echo "✅ .env file created. Please add your OpenAI API key."
else
    echo "✅ .env file already exists"
fi

# Create directories if they don't exist
echo "📁 Creating necessary directories..."
mkdir -p input output

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  ffmpeg is not installed. Audio processing may not work properly."
    echo "📥 To install ffmpeg:"
    echo "   - Windows: Download from https://www.gyan.dev/ffmpeg/builds/"
    echo "   - macOS: brew install ffmpeg"
    echo "   - Ubuntu/Debian: sudo apt install ffmpeg"
    echo "   - CentOS/RHEL: sudo yum install ffmpeg"
else
    echo "✅ ffmpeg found: $(ffmpeg -version | head -n1)"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Add your OpenAI API key to the .env file"
echo "2. Place audio files in the 'input' folder"
echo "3. Run the transcription script:"
echo "   - For all files: python transcribe.py"
echo "   - For specific file: python transcribe.py --file input/your_file.mp3"
echo ""
echo "📖 Usage examples:"
echo "   python transcribe.py                    # Process all files in input folder"
echo "   python transcribe.py --file input/audio.mp3  # Process specific file"
echo ""
echo "📁 Project structure:"
echo "   ├── input/          # Place audio files here"
echo "   ├── output/         # Transcriptions will be saved here"
echo "   ├── transcribe.py   # Main transcription script"
echo "   ├── setup.sh        # This setup script"
echo "   └── .env           # API configuration" 