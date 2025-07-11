# Voice Transcriber Setup Script for Windows PowerShell
# This script sets up the environment for the voice transcription project

Write-Host "🎤 Voice Transcriber Setup Script" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed. Please install Python first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if pip is installed
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✅ pip found" -ForegroundColor Green
} catch {
    Write-Host "❌ pip is not installed. Please install pip first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Install required packages
Write-Host "📚 Installing required packages..." -ForegroundColor Yellow
pip install -r requirements.txt

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "🔑 Creating .env file template..." -ForegroundColor Yellow
    @"
# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here
"@ | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "✅ .env file created. Please add your OpenAI API key." -ForegroundColor Green
} else {
    Write-Host "✅ .env file already exists" -ForegroundColor Green
}

# Create directories if they don't exist
Write-Host "📁 Creating necessary directories..." -ForegroundColor Yellow
if (-not (Test-Path "input")) { New-Item -ItemType Directory -Name "input" }
if (-not (Test-Path "output")) { New-Item -ItemType Directory -Name "output" }

# Check if ffmpeg is installed
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "✅ ffmpeg found: $ffmpegVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠️  ffmpeg is not installed. Audio processing may not work properly." -ForegroundColor Yellow
    Write-Host "📥 To install ffmpeg:" -ForegroundColor Cyan
    Write-Host "   - Download from https://www.gyan.dev/ffmpeg/builds/" -ForegroundColor Cyan
    Write-Host "   - Extract to C:\ffmpeg" -ForegroundColor Cyan
    Write-Host "   - Add C:\ffmpeg\bin to your PATH environment variable" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "🎉 Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Add your OpenAI API key to the .env file" -ForegroundColor White
Write-Host "2. Place audio files in the 'input' folder" -ForegroundColor White
Write-Host "3. Run the transcription script:" -ForegroundColor White
Write-Host "   - For all files: python transcribe.py" -ForegroundColor White
Write-Host "   - For specific file: python transcribe.py --file input/your_file.mp3" -ForegroundColor White
Write-Host ""
Write-Host "📖 Usage examples:" -ForegroundColor Cyan
Write-Host "   python transcribe.py                    # Process all files in input folder" -ForegroundColor White
Write-Host "   python transcribe.py --file input/audio.mp3  # Process specific file" -ForegroundColor White
Write-Host ""
Write-Host "📁 Project structure:" -ForegroundColor Cyan
Write-Host "   ├── input/          # Place audio files here" -ForegroundColor White
Write-Host "   ├── output/         # Transcriptions will be saved here" -ForegroundColor White
Write-Host "   ├── transcribe.py   # Main transcription script" -ForegroundColor White
Write-Host "   ├── setup.ps1       # This setup script" -ForegroundColor White
Write-Host "   └── .env           # API configuration" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to continue" 