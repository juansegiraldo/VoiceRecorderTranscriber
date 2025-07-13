@echo off
REM M4A to MP3 Converter Batch Wrapper
REM This script provides a convenient way to convert M4A files to MP3 on Windows

setlocal enabledelayedexpansion

REM Check if input path is provided
if "%~1"=="" (
    echo Error: Input path is required.
    echo Usage: convert_m4a_to_mp3.bat ^<input_path^> [output_path] [bitrate]
    echo.
    echo Examples:
    echo   convert_m4a_to_mp3.bat input.m4a
    echo   convert_m4a_to_mp3.bat input.m4a output.mp3
    echo   convert_m4a_to_mp3.bat input_folder/
    echo   convert_m4a_to_mp3.bat input_folder/ output_folder/ 320k
    exit /b 1
)

set INPUT_PATH=%~1
set OUTPUT_PATH=%~2
set BITRATE=%~3

REM Set default bitrate if not provided
if "%BITRATE%"=="" set BITRATE=192k

echo M4A to MP3 Converter
echo ====================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not available. Please install Python and try again.
    exit /b 1
)

REM Check if required packages are installed
python -c "import pydub" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install pydub
    if errorlevel 1 (
        echo Error: Failed to install required packages.
        exit /b 1
    )
)

REM Build the Python command
set PYTHON_SCRIPT=convert_m4a_to_mp3.py
set PYTHON_ARGS=%INPUT_PATH%

if not "%OUTPUT_PATH%"=="" (
    set PYTHON_ARGS=%PYTHON_ARGS% %OUTPUT_PATH%
)

REM Add bitrate if not default
if not "%BITRATE%"=="192k" (
    set PYTHON_ARGS=%PYTHON_ARGS% --bitrate %BITRATE%
)

echo Running conversion...
echo Command: python %PYTHON_SCRIPT% %PYTHON_ARGS%

REM Execute the Python script
python %PYTHON_SCRIPT% %PYTHON_ARGS%
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% equ 0 (
    echo Conversion completed successfully!
) else (
    echo Conversion failed with exit code: %EXIT_CODE%
)

exit /b %EXIT_CODE% 