# M4A to MP3 Converter

This directory contains scripts to convert M4A audio files to MP3 format. The converter uses the `pydub` library which is already included in the project dependencies.

## Files

- `convert_m4a_to_mp3.py` - Main Python script for conversion
- `convert_m4a_to_mp3.ps1` - PowerShell wrapper script (Windows)
- `convert_m4a_to_mp3.bat` - Batch file wrapper (Windows)

## Prerequisites

- Python 3.6 or higher
- `pydub` library (already in requirements.txt)
- FFmpeg (required by pydub for audio conversion)

## Installation

The required packages are already included in the project's `requirements.txt`. If you haven't installed them yet:

```powershell
pip install -r requirements.txt
```

## Usage

### Using the Python Script Directly

```bash
# Convert a single file
python convert_m4a_to_mp3.py input.m4a

# Convert with custom output path
python convert_m4a_to_mp3.py input.m4a output.mp3

# Convert with custom bitrate
python convert_m4a_to_mp3.py input.m4a --bitrate 320k

# Convert all M4A files in a directory
python convert_m4a_to_mp3.py input_folder/

# Convert directory with custom output directory
python convert_m4a_to_mp3.py input_folder/ output_folder/

# Enable verbose logging
python convert_m4a_to_mp3.py input.m4a --verbose
```

### Using PowerShell (Windows)

```powershell
# Convert a single file
.\convert_m4a_to_mp3.ps1 "input.m4a"

# Convert with custom output path
.\convert_m4a_to_mp3.ps1 "input.m4a" "output.mp3"

# Convert with custom bitrate
.\convert_m4a_to_mp3.ps1 "input.m4a" -Bitrate "320k"

# Convert all M4A files in a directory
.\convert_m4a_to_mp3.ps1 "input_folder/"

# Enable verbose logging
.\convert_m4a_to_mp3.ps1 "input.m4a" -Verbose
```

### Using Batch File (Windows)

```cmd
# Convert a single file
convert_m4a_to_mp3.bat input.m4a

# Convert with custom output path
convert_m4a_to_mp3.bat input.m4a output.mp3

# Convert with custom bitrate
convert_m4a_to_mp3.bat input.m4a output.mp3 320k

# Convert all M4A files in a directory
convert_m4a_to_mp3.bat input_folder/
```

## Features

- **Single File Conversion**: Convert individual M4A files to MP3
- **Batch Directory Conversion**: Convert all M4A files in a directory at once
- **Custom Output Paths**: Specify custom output locations
- **Custom Bitrates**: Choose MP3 bitrate (default: 192k)
- **Verbose Logging**: Detailed logging for debugging
- **Error Handling**: Comprehensive error handling and reporting
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Supported Bitrates

Common MP3 bitrates you can use:
- `64k` - Low quality, small file size
- `128k` - Standard quality
- `192k` - High quality (default)
- `256k` - Very high quality
- `320k` - Maximum quality

## Examples

### Convert a single file
```powershell
.\convert_m4a_to_mp3.ps1 "C:\Users\username\Desktop\recording.m4a"
```

### Convert with custom output and bitrate
```powershell
.\convert_m4a_to_mp3.ps1 "recording.m4a" "converted_recording.mp3" -Bitrate "320k"
```

### Convert all files in a folder
```powershell
.\convert_m4a_to_mp3.ps1 "C:\Recordings\" "C:\Converted\"
```

## Troubleshooting

### Common Issues

1. **"Python not found"**
   - Make sure Python is installed and added to your PATH
   - Try running `python --version` to verify

2. **"pydub not found"**
   - Install the required packages: `pip install pydub`
   - Or install all project dependencies: `pip install -r requirements.txt`

3. **"FFmpeg not found"**
   - pydub requires FFmpeg for audio conversion
   - Download and install FFmpeg from https://ffmpeg.org/
   - Add FFmpeg to your system PATH

4. **"Permission denied"**
   - Make sure you have write permissions to the output directory
   - Try running as administrator if needed

### Getting Help

To see all available options:
```bash
python convert_m4a_to_mp3.py --help
```

## Integration with VoiceTranscriber

This converter is designed to work seamlessly with the VoiceTranscriber project. After converting M4A files to MP3, you can use them with the transcription scripts:

1. Convert your M4A files to MP3 using this converter
2. Place the MP3 files in the `input/` directory
3. Run the transcription scripts as usual

## License

This converter is part of the VoiceTranscriber project and follows the same license terms. 