import argparse
import os
import shutil
import sys
from pathlib import Path

import whisper

MEDIA_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv"
}


def ensure_ffmpeg_in_path():
    if shutil.which("ffmpeg"):
        return
    winget_dir = Path.home() / "AppData/Local/Microsoft/WinGet"
    if winget_dir.exists():
        ffmpeg_execs = list(winget_dir.rglob("ffmpeg.exe"))
        if ffmpeg_execs:
            ffmpeg_bin_dir = str(ffmpeg_execs[0].parent)
            os.environ["PATH"] = ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")
            print(f"Added FFmpeg to PATH: {ffmpeg_bin_dir}")
            return


ensure_ffmpeg_in_path()


def is_media_file(path: Path) -> bool:

    return path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS


def transcribe_file(file_path: Path, model: whisper.Whisper, output_dir: Path) -> Path:
    print(f"\nTranscribing: {file_path.name}")
    result = model.transcribe(str(file_path))
    transcript = result["text"].strip()
    language = result.get("language", "unknown")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = file_path.stem
    output_file = output_dir / f"{file_stem}.txt"
    output_file.write_text(transcript, encoding="utf-8")

    lang_file = output_dir / f"{file_stem}.lang"
    lang_file.write_text(language, encoding="utf-8")

    print(f"Transcript saved to: {output_file}")
    print(f"Detected language: {language}")
    return output_file


def transcribe_media(input_path: str, model_size: str = "medium", output_dir: str = "Output/transcripts") -> list[Path]:
    target = Path(input_path)
    if not target.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)

    output_path = Path(output_dir)

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted([f for f in target.rglob("*") if is_media_file(f)])
        if not files:
            print(f"No supported media files found in: {input_path}")
            return []
        print(f"Found {len(files)} media file(s) in {input_path}")
    else:
        print(f"Error: Invalid path: {input_path}")
        sys.exit(1)

    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)

    generated = []
    for f in files:
        txt_path = transcribe_file(f, model, output_path)
        generated.append(txt_path)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video files using OpenAI Whisper.")
    parser.add_argument("input", help="Path to audio/video file or directory")
    parser.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--output-dir",
        default="Output/transcripts",
        help="Directory to save transcripts (default: Output/transcripts)",
    )
    args = parser.parse_args()

    transcribe_media(args.input, args.model, args.output_dir)


if __name__ == "__main__":
    main()

