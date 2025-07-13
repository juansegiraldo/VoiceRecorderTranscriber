@echo off
REM Voice Transcriber Setup Script for Windows
REM This script sets up the environment for the voice transcription project

echo 🎤 Voice Transcriber Setup Script
echo ==================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python first.
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Please install pip first.
    pause
    exit /b 1
)

echo ✅ pip found

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install required packages
echo 📚 Installing required packages...
pip install -r requirements.txt

REM Check if .env file exists
if not exist ".env" (
    echo 🔑 Creating .env file template...
    (
        echo # OpenAI API Configuration
        echo # Get your API key from: https://platform.openai.com/api-keys
        echo OPENAI_API_KEY=your_openai_api_key_here
    ) > .env
    echo ✅ .env file created. Please add your OpenAI API key.
) else (
    echo ✅ .env file already exists
)

REM Create directories if they don't exist
echo 📁 Creating necessary directories...
if not exist "input" mkdir input
if not exist "output" mkdir output

REM Check if ffmpeg is installed
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  ffmpeg is not installed. Audio processing may not work properly.
    echo 📥 To install ffmpeg:
    echo    - Download from https://www.gyan.dev/ffmpeg/builds/
    echo    - Extract to C:\ffmpeg
    echo    - Add C:\ffmpeg\bin to your PATH environment variable
) else (
    echo ✅ ffmpeg found
)

echo.
echo 🎉 Setup complete!
echo.
echo 📋 Next steps:
echo 1. Add your OpenAI API key to the .env file
echo 2. Place audio files in the 'input' folder
echo 3. Run the transcription script:
echo    - For all files: python transcribe.py
echo    - For specific file: python transcribe.py --file input/your_file.mp3
echo.
echo 📖 Usage examples:
echo    python transcribe.py                    # Process all files in input folder
echo    python transcribe.py --file input/audio.mp3  # Process specific file
echo.
echo 📁 Project structure:
echo    ├── input/          # Place audio files here
echo    ├── output/         # Transcriptions will be saved here
echo    ├── transcribe.py   # Main transcription script
echo    ├── setup.bat       # This setup script
echo    └── .env           # API configuration
echo.
pause 