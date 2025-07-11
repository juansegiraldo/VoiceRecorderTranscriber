import argparse
from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path
import tempfile
import shutil
from pydub import AudioSegment
import math

# Load environment variables from .env file
load_dotenv()


def split_audio_file(file_path: str, max_size_mb: int = 24) -> list[str]:
    """Split a large audio file into smaller chunks that fit within API limits."""
    # Convert MB to bytes (leaving some buffer)
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # If file is small enough, return original path
    if file_size <= max_size_bytes:
        return [file_path]
    
    # Calculate how many chunks we need
    num_chunks = math.ceil(file_size / max_size_bytes)
    chunk_duration = len(audio) // num_chunks
    
    # Create temporary directory for chunks
    temp_dir = tempfile.mkdtemp()
    chunk_paths = []
    
    try:
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = start_time + chunk_duration if i < num_chunks - 1 else len(audio)
            
            # Extract chunk
            chunk = audio[start_time:end_time]
            
            # Save chunk to temporary file
            chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.mp3")
            chunk.export(chunk_path, format="mp3")
            chunk_paths.append(chunk_path)
            
        return chunk_paths
    except Exception as e:
        # Clean up temp directory on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


def transcribe_file(path: str) -> str:
    """Upload an audio file to OpenAI and return the transcription text."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)

    with open(path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text


def transcribe_large_file(file_path: str) -> str:
    """Transcribe a large audio file by splitting it into chunks and combining results."""
    print(f"Archivo grande detectado. Dividiendo en chunks...")
    
    # Split the file into chunks
    chunk_paths = split_audio_file(file_path)
    
    if len(chunk_paths) == 1:
        # File is small enough, process normally
        return transcribe_file(file_path)
    
    print(f"Archivo dividido en {len(chunk_paths)} chunks")
    
    # Process each chunk
    transcriptions = []
    temp_dir = os.path.dirname(chunk_paths[0]) if len(chunk_paths) > 1 else None
    
    try:
        for i, chunk_path in enumerate(chunk_paths, 1):
            print(f"Procesando chunk {i}/{len(chunk_paths)}...")
            try:
                chunk_transcription = transcribe_file(chunk_path)
                transcriptions.append(chunk_transcription)
                print(f"✓ Chunk {i} procesado")
            except Exception as e:
                print(f"✗ Error procesando chunk {i}: {str(e)}")
                # Continue with other chunks even if one fails
                transcriptions.append(f"[Error en chunk {i}: {str(e)}]")
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Combine all transcriptions
    combined_transcription = " ".join(transcriptions)
    return combined_transcription


def process_input_folder():
    """Process all audio files in the input folder and save transcriptions to output folder."""
    input_dir = Path("input")
    output_dir = Path("output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Supported audio file extensions
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
    
    # Get all audio files from input directory
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))
        audio_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        print("No se encontraron archivos de audio en la carpeta 'input'")
        print(f"Extensiones soportadas: {', '.join(audio_extensions)}")
        return
    
    print(f"Se encontraron {len(audio_files)} archivo(s) de audio para procesar:")
    for file in audio_files:
        print(f"  - {file.name}")
    
    print("\nProcesando archivos...")
    
    for audio_file in audio_files:
        try:
            print(f"\nTranscribiendo: {audio_file.name}")
            text = transcribe_large_file(str(audio_file))
            
            # Create output filename
            output_filename = audio_file.stem + '_transcript.txt'
            output_path = output_dir / output_filename
            
            # Save transcription
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"✓ Transcripción guardada en: {output_path}")
            
        except Exception as e:
            print(f"✗ Error procesando {audio_file.name}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI's Whisper API")
    parser.add_argument("--file", help="Path to a specific audio file (optional)")
    args = parser.parse_args()

    if args.file:
        # Process single file
        if not os.path.exists(args.file):
            print(f"Error: El archivo '{args.file}' no existe")
            return
        
        text = transcribe_large_file(args.file)
        print(text)
        
        # Save to output folder
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Get filename from path
        audio_file = Path(args.file)
        output_filename = audio_file.stem + '_transcript.txt'
        output_path = output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"\nTranscripción guardada en: {output_path}")
    else:
        # Process all files in input folder
        process_input_folder()


if __name__ == "__main__":
    main()

