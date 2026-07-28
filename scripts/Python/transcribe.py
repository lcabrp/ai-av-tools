import argparse
import os
import sys
from pathlib import Path

import whisper


def transcribe_audio(audio_path: str, model_size: str = "base") -> tuple[str, str]:
    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {audio_file.name}")
    result = model.transcribe(str(audio_file))

    return result["text"], result.get("language", "unknown")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI Whisper.")
    parser.add_argument("audio", help="Path to the audio file")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--output-dir",
        default="Output/transcripts",
        help="Directory to save transcripts (default: Output/transcripts)",
    )
    args = parser.parse_args()

    transcript, language = transcribe_audio(args.audio, args.model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_name = Path(args.audio).stem
    output_file = output_dir / f"{audio_name}.txt"
    output_file.write_text(transcript, encoding="utf-8")

    lang_file = output_dir / f"{audio_name}.lang"
    lang_file.write_text(language, encoding="utf-8")

    print(f"Transcript saved to: {output_file}")
    print(f"Detected language: {language}")


if __name__ == "__main__":
    main()
