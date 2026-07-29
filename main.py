import argparse
import sys
from pathlib import Path

# Add project root to sys.path if needed
sys.path.insert(0, str(Path(__file__).parent))

from scripts.Python.summarize import summarize_transcripts
from scripts.Python.transcribe import transcribe_media


def process_pipeline(
    input_path: str,
    whisper_model: str = "base",
    provider: str = "auto",
    openai_model: str = "gpt-4o-mini",
    anthropic_model: str = "claude-sonnet-4-6",
    gemini_model: str = "gemini-flash-latest",
    language: str | None = None,
    transcripts_dir: str = "Output/transcripts",
    output_dir: str = "Output",
):
    print("=== Step 1: Transcribing audio/video ===")
    transcript_files = transcribe_media(
        input_path=input_path,
        model_size=whisper_model,
        output_dir=transcripts_dir,
    )

    if not transcript_files:
        print("No transcripts generated. Exiting pipeline.")
        return

    print("\n=== Step 2: Summarizing transcripts ===")
    for tf in transcript_files:
        summarize_transcripts(
            input_path=str(tf),
            output=output_dir,
            language=language,
            provider=provider,
            openai_model=openai_model,
            anthropic_model=anthropic_model,
            gemini_model=gemini_model,
        )

    print("\n=== Pipeline Complete ===")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end media transcription and summarization pipeline."
    )
    parser.add_argument("input", help="Path to audio/video file or directory")
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for transcription (default: base)",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "ollama", "openai", "anthropic", "gemini"],
        help="LLM provider for summary: auto (ollama -> anthropic -> gemini -> openai), ollama, openai, anthropic, or gemini (default: auto)",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini, e.g. gpt-4o, o3-mini)",
    )
    parser.add_argument(
        "--anthropic-model",
        default="claude-sonnet-4-6",
        help="Anthropic model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-flash-latest",
        help="Gemini model to use (default: gemini-flash-latest)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Target output language for summary (default: auto-detect or English)",
    )
    parser.add_argument(
        "--transcripts-dir",
        default="Output/transcripts",
        help="Directory to save generated transcripts (default: Output/transcripts)",
    )
    parser.add_argument(
        "--output-dir",
        default="Output",
        help="Directory to save summary markdown files (default: Output)",
    )

    args = parser.parse_args()
    process_pipeline(
        input_path=args.input,
        whisper_model=args.whisper_model,
        provider=args.provider,
        openai_model=args.openai_model,
        anthropic_model=args.anthropic_model,
        gemini_model=args.gemini_model,
        language=args.language,
        transcripts_dir=args.transcripts_dir,
        output_dir=args.output_dir,
    )




if __name__ == "__main__":
    main()

