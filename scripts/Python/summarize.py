import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def summarize_transcript(transcript_path: str, output_path: str, language: str = "english", provider: str = "ollama") -> None:
    transcript_file = Path(transcript_path)
    if not transcript_file.exists():
        print(f"Error: File not found: {transcript_path}")
        sys.exit(1)

    transcript = transcript_file.read_text(encoding="utf-8")

    if provider == "openai":
        client = OpenAI()
        model = "gpt-4o-mini"
    else:
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        model = "gemma4:e4b"

    print(f"Generating summary with {provider} ({model})...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a helpful assistant that summarizes transcripts. "
                    f"Extract the main points and present them as a clean markdown document "
                    f"with a title, key topics, and bullet points for each main idea. "
                    f"Write the entire summary in {language}."
                ),
            },
            {
                "role": "user",
                "content": f"Please summarize the following transcript into a markdown file with the main points:\n\n{transcript}",
            },
        ],
    )

    summary = response.choices[0].message.content

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(summary, encoding="utf-8")

    print(f"Summary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Summarize a transcript into a markdown file.")
    parser.add_argument("transcript", help="Path to the transcript file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown file path (default: Output/<transcript_name>_summary.md)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Output language (default: auto-detect from transcript)",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "openai"],
        help="LLM provider (default: ollama)",
    )
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        transcript_name = Path(args.transcript).stem
        output_path = f"Output/{transcript_name}_summary.md"

    if args.language:
        language = args.language
    else:
        lang_file = Path(args.transcript).with_suffix(".lang")
        if lang_file.exists():
            language = lang_file.read_text(encoding="utf-8").strip()
        else:
            language = "english"

    summarize_transcript(args.transcript, output_path, language, args.provider)


if __name__ == "__main__":
    main()
