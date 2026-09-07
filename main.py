import argparse
import sys
from pathlib import Path

# Add project root to sys.path if needed
sys.path.insert(0, str(Path(__file__).parent))

from scripts.Python.summarize import summarize_transcripts
from scripts.Python.transcribe import transcribe_media
from scripts.Python.youtube_transcript import fetch_youtube_transcript, is_youtube_input


def process_pipeline(
    input_path: str,
    whisper_model: str = "medium",
    provider: str = "auto",
    openai_model: str = "gpt-5.6-luna",
    anthropic_model: str = "claude-sonnet-4-6",
    gemini_model: str = "gemini-flash-latest",
    language: str | None = None,
    transcripts_dir: str = "Output/transcripts",
    output_dir: str = "Output",
    use_chunking: bool = False,
    chunk_chars: int = 3000,
    gemini_chunk_chars: int = 15000,
):
    if is_youtube_input(input_path):
        print("=== Step 1: Fetching YouTube transcript ===")
        transcript_files = fetch_youtube_transcript(
            url_or_id=input_path,
            output_dir=transcripts_dir,
            language=language,
        )
    else:
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
            use_chunking=use_chunking,
            chunk_chars=chunk_chars,
            gemini_chunk_chars=gemini_chunk_chars,
        )

    print("\n=== Pipeline Complete ===")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end media or YouTube video transcription and summarization pipeline."
    )
    parser.add_argument("input", help="Path to audio/video file, directory, or YouTube URL/ID")
    parser.add_argument(
        "--whisper-model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for local media transcription (default: medium)",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "ollama", "openai", "anthropic", "gemini"],
        help="LLM provider for summary: auto (gemini -> anthropic -> openai -> ollama), ollama, openai, anthropic, or gemini (default: auto)",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-5.6-luna",
        help="OpenAI model to use (default: gpt-5.6-luna, e.g. gpt-5.6-terra, gpt-5.6-sol)",
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
        help="Target output language for summary / transcript (default: auto-detect or English)",
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
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Use map-reduce chunked summarization for the local Ollama fallback, to avoid detail loss on "
             "long transcripts with small/low-VRAM models (recommended on 4-8GB GPUs). Gemini already "
             "auto-chunks long transcripts regardless of this flag.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=3000,
        help="Max characters per chunk when --chunked is used (default: 3000; lower for less VRAM)",
    )
    parser.add_argument(
        "--gemini-chunk-chars",
        type=int,
        default=15000,
        help="Chunk size threshold for Gemini: transcripts longer than this are automatically "
             "summarized via map-reduce chunking for better detail retention (default: 15000)",
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
        use_chunking=args.chunked,
        chunk_chars=args.chunk_chars,
        gemini_chunk_chars=args.gemini_chunk_chars,
    )


if __name__ == "__main__":
    main()
