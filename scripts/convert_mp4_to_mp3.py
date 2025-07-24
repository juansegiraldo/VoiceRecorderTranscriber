#!/usr/bin/env python3
"""
MP4 to MP3 Converter Script

This script converts MP4 video files to MP3 audio format using the pydub library.
It can process single files or entire directories.

Usage:
    python convert_mp4_to_mp3.py <input_path> [output_path]
    
Examples:
    python convert_mp4_to_mp3.py input.mp4
    python convert_mp4_to_mp3.py input.mp4 output.mp3
    python convert_mp4_to_mp3.py input_folder/
    python convert_mp4_to_mp3.py input_folder/ output_folder/

Interactive Mode:
    python convert_mp4_to_mp3.py
    (Automatically processes MP4 files from input/ to output/)
"""

import os
import sys
import argparse
from pathlib import Path
from pydub import AudioSegment
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_mp4_to_mp3(input_path, output_path=None, bitrate="192k"):
    """
    Convert a single MP4 file to MP3 format.
    
    Args:
        input_path (str): Path to the input MP4 file
        output_path (str, optional): Path for the output MP3 file. If None, 
                                   will use the same name with .mp3 extension
        bitrate (str): MP3 bitrate (default: "192k")
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        input_file = Path(input_path)
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_path}")
            return False
            
        if not input_file.suffix.lower() == '.mp4':
            logger.warning(f"File {input_path} doesn't have .mp4 extension")
        
        # Determine output path
        if output_path is None:
            output_file = input_file.with_suffix('.mp3')
        else:
            output_file = Path(output_path)
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting {input_path} to {output_file}")
        
        # Load the video file and extract audio
        audio = AudioSegment.from_file(str(input_file), format="mp4")
        
        # Export as MP3
        audio.export(str(output_file), format="mp3", bitrate=bitrate)
        
        logger.info(f"Successfully converted {input_path} to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting {input_path}: {str(e)}")
        return False

def convert_directory(input_dir, output_dir=None, bitrate="192k"):
    """
    Convert all MP4 files in a directory to MP3 format.
    
    Args:
        input_dir (str): Path to the input directory
        output_dir (str, optional): Path for the output directory. If None,
                                  will use the same directory
        bitrate (str): MP3 bitrate (default: "192k")
    
    Returns:
        tuple: (success_count, total_count)
    """
    input_path = Path(input_dir)
    
    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return 0, 0
    
    # Find all MP4 files
    mp4_files = list(input_path.glob("*.mp4"))
    
    if not mp4_files:
        logger.warning(f"No MP4 files found in {input_dir}")
        return 0, 0
    
    logger.info(f"Found {len(mp4_files)} MP4 files to convert")
    
    success_count = 0
    total_count = len(mp4_files)
    
    for mp4_file in mp4_files:
        if output_dir is None:
            output_file = mp4_file.with_suffix('.mp3')
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"{mp4_file.stem}.mp3"
        
        if convert_mp4_to_mp3(str(mp4_file), str(output_file), bitrate):
            success_count += 1
    
    logger.info(f"Conversion complete: {success_count}/{total_count} files converted successfully")
    return success_count, total_count

def run_interactive_mode():
    """
    Run the script in interactive mode, automatically processing MP4 files
    from the input/ directory to the output/ directory.
    """
    logger.info("Running in interactive mode...")
    
    # Define default directories
    input_dir = Path("input")
    output_dir = Path("output")
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input directory '{input_dir}' not found!")
        logger.info("Please create an 'input' directory and place your MP4 files there.")
        return False
    
    # Find all MP4 files in input directory
    mp4_files = list(input_dir.glob("*.mp4"))
    
    if not mp4_files:
        logger.warning(f"No MP4 files found in {input_dir}")
        logger.info("Please place MP4 files in the 'input' directory.")
        return False
    
    logger.info(f"Found {len(mp4_files)} MP4 files to convert:")
    for file in mp4_files:
        logger.info(f"  - {file.name}")
    
    # Convert all files
    success_count, total_count = convert_directory(str(input_dir), str(output_dir))
    
    if success_count > 0:
        logger.info(f"✅ Successfully converted {success_count} files to the 'output' directory!")
    else:
        logger.error("❌ No files were converted successfully.")
    
    return success_count > 0

def main():
    # Check if no arguments were provided (interactive mode)
    if len(sys.argv) == 1:
        success = run_interactive_mode()
        sys.exit(0 if success else 1)
    
    # Command line mode
    parser = argparse.ArgumentParser(
        description="Convert MP4 video files to MP3 audio format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "input_path",
        help="Input MP4 file or directory containing MP4 files"
    )
    
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Output MP3 file or directory (optional)"
    )
    
    parser.add_argument(
        "--bitrate",
        default="192k",
        help="MP3 bitrate (default: 192k)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {args.input_path}")
        sys.exit(1)
    
    if input_path.is_file():
        # Convert single file
        success = convert_mp4_to_mp3(args.input_path, args.output_path, args.bitrate)
        sys.exit(0 if success else 1)
    
    elif input_path.is_dir():
        # Convert directory
        success_count, total_count = convert_directory(args.input_path, args.output_path, args.bitrate)
        sys.exit(0 if success_count == total_count else 1)
    
    else:
        logger.error(f"Input path is neither a file nor directory: {args.input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main() 