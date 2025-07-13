#!/usr/bin/env python3
"""
M4A to MP3 Converter Script

This script converts M4A audio files to MP3 format using the pydub library.
It can process single files or entire directories.

Usage:
    python convert_m4a_to_mp3.py <input_path> [output_path]
    
Examples:
    python convert_m4a_to_mp3.py input.m4a
    python convert_m4a_to_mp3.py input.m4a output.mp3
    python convert_m4a_to_mp3.py input_folder/
    python convert_m4a_to_mp3.py input_folder/ output_folder/

Interactive Mode:
    python convert_m4a_to_mp3.py
    (Automatically processes M4A files from m4a_input/ to mp3_output/)
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

def convert_m4a_to_mp3(input_path, output_path=None, bitrate="192k"):
    """
    Convert a single M4A file to MP3 format.
    
    Args:
        input_path (str): Path to the input M4A file
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
            
        if not input_file.suffix.lower() == '.m4a':
            logger.warning(f"File {input_path} doesn't have .m4a extension")
        
        # Determine output path
        if output_path is None:
            output_file = input_file.with_suffix('.mp3')
        else:
            output_file = Path(output_path)
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting {input_path} to {output_file}")
        
        # Load the audio file
        audio = AudioSegment.from_file(str(input_file), format="m4a")
        
        # Export as MP3
        audio.export(str(output_file), format="mp3", bitrate=bitrate)
        
        logger.info(f"Successfully converted {input_path} to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting {input_path}: {str(e)}")
        return False

def convert_directory(input_dir, output_dir=None, bitrate="192k"):
    """
    Convert all M4A files in a directory to MP3 format.
    
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
    
    # Find all M4A files
    m4a_files = list(input_path.glob("*.m4a"))
    
    if not m4a_files:
        logger.warning(f"No M4A files found in {input_dir}")
        return 0, 0
    
    logger.info(f"Found {len(m4a_files)} M4A files to convert")
    
    success_count = 0
    total_count = len(m4a_files)
    
    for m4a_file in m4a_files:
        if output_dir is None:
            output_file = m4a_file.with_suffix('.mp3')
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"{m4a_file.stem}.mp3"
        
        if convert_m4a_to_mp3(str(m4a_file), str(output_file), bitrate):
            success_count += 1
    
    logger.info(f"Conversion complete: {success_count}/{total_count} files converted successfully")
    return success_count, total_count

def run_interactive_mode():
    """
    Run the script in interactive mode, automatically processing M4A files
    from the m4a_input/ directory to the mp3_output/ directory.
    """
    logger.info("Running in interactive mode...")
    
    # Define default directories
    input_dir = Path("m4a_input")
    output_dir = Path("mp3_output")
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input directory '{input_dir}' not found!")
        logger.info("Please create a 'm4a_input' directory and place your M4A files there.")
        return False
    
    # Find all M4A files in input directory
    m4a_files = list(input_dir.glob("*.m4a"))
    
    if not m4a_files:
        logger.warning(f"No M4A files found in {input_dir}")
        logger.info("Please place M4A files in the 'm4a_input' directory.")
        return False
    
    logger.info(f"Found {len(m4a_files)} M4A files to convert:")
    for file in m4a_files:
        logger.info(f"  - {file.name}")
    
    # Convert all files
    success_count, total_count = convert_directory(str(input_dir), str(output_dir))
    
    if success_count > 0:
        logger.info(f"✅ Successfully converted {success_count} files to the 'mp3_output' directory!")
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
        description="Convert M4A files to MP3 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "input_path",
        help="Input M4A file or directory containing M4A files"
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
        success = convert_m4a_to_mp3(args.input_path, args.output_path, args.bitrate)
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