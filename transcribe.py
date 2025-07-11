import argparse
import openai
import os


def transcribe_file(path: str) -> str:
    """Upload an audio file to OpenAI and return the transcription text."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    openai.api_key = api_key

    with open(path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file using OpenAI's Whisper API")
    parser.add_argument("file", help="Path to the audio file")
    args = parser.parse_args()

    text = transcribe_file(args.file)
    print(text)


if __name__ == "__main__":
    main()

