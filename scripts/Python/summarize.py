import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_PRIMARY_LOCAL = "gemma4:e4b"
DEFAULT_SECONDARY_LOCAL = "gemma4:e2b"
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")



def generate_with_openai_compatible(client: OpenAI, model: str, transcript: str, language: str) -> str:
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
    return response.choices[0].message.content


def generate_with_anthropic(model: str, transcript: str, language: str) -> str:
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment or .env file.")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=(
            f"You are a helpful assistant that summarizes transcripts. "
            f"Extract the main points and present them as a clean markdown document "
            f"with a title, key topics, and bullet points for each main idea. "
            f"Write the entire summary in {language}."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Please summarize the following transcript into a markdown file with the main points:\n\n{transcript}",
            }
        ],
    )
    return response.content[0].text


def generate_with_gemini(model: str, transcript: str, language: str) -> str:
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY / GOOGLE_API_KEY not found in environment or .env file.")
    client = genai.Client(api_key=api_key)
    prompt = (
        f"You are a helpful assistant that summarizes transcripts. "
        f"Extract the main points and present them as a clean markdown document "
        f"with a title, key topics, and bullet points for each main idea. "
        f"Write the entire summary in {language}.\n\n"
        f"Please summarize the following transcript into a markdown file with the main points:\n\n{transcript}"
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text


def summarize_single_transcript(
    transcript_path: Path,
    output_path: Path,
    language: str | None = None,
    provider: str = "auto",
    primary_local_model: str = DEFAULT_PRIMARY_LOCAL,
    secondary_local_model: str = DEFAULT_SECONDARY_LOCAL,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> bool:
    if not transcript_path.exists():
        print(f"Error: Transcript file not found: {transcript_path}")
        return False

    transcript = transcript_path.read_text(encoding="utf-8")
    if not transcript.strip():
        print(f"Warning: Transcript {transcript_path.name} is empty. Skipping.")
        return False

    if not language:
        lang_file = transcript_path.with_suffix(".lang")
        if lang_file.exists():
            language = lang_file.read_text(encoding="utf-8").strip()
        else:
            language = "english"

    summary = None

    if provider in ("auto", "ollama"):
        ollama_client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
        )
        try:
            print(f"Generating summary with local Ollama ({primary_local_model}) for {transcript_path.name}...")
            summary = generate_with_openai_compatible(ollama_client, primary_local_model, transcript, language)
        except Exception as e:
            print(f"Ollama primary model ({primary_local_model}) failed: {e}")
            if provider == "auto":
                try:
                    print(f"Falling back to secondary local Ollama model ({secondary_local_model})...")
                    summary = generate_with_openai_compatible(ollama_client, secondary_local_model, transcript, language)
                except Exception as e2:
                    print(f"Ollama secondary model ({secondary_local_model}) failed: {e2}")

    if summary is None and provider in ("auto", "anthropic"):
        try:
            print(f"Generating summary with Anthropic ({anthropic_model})...")
            summary = generate_with_anthropic(anthropic_model, transcript, language)
        except Exception as e:
            print(f"Anthropic API failed: {e}")

    if summary is None and provider in ("auto", "gemini"):
        try:
            print(f"Generating summary with Gemini ({gemini_model})...")
            summary = generate_with_gemini(gemini_model, transcript, language)
        except Exception as e:
            print(f"Gemini API failed: {e}")

    if summary is None and provider in ("auto", "openai"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OpenAI API key not found in environment or .env file.")
        else:
            try:
                print(f"Generating summary with OpenAI ({openai_model})...")
                openai_client = OpenAI(api_key=api_key)
                summary = generate_with_openai_compatible(openai_client, openai_model, transcript, language)
            except Exception as e:
                print(f"OpenAI API failed: {e}")

    if summary is None:
        print(f"Error: All summary providers failed for {transcript_path.name}.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    print(f"Summary saved to: {output_path}")
    return True


def summarize_transcripts(
    input_path: str,
    output: str | None = None,
    language: str | None = None,
    provider: str = "auto",
    openai_model: str = DEFAULT_OPENAI_MODEL,
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> list[Path]:
    target = Path(input_path)
    if not target.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)

    if target.is_file():
        transcript_files = [target]
    elif target.is_dir():
        transcript_files = sorted(list(target.glob("*.txt")))
        if not transcript_files:
            print(f"No .txt transcript files found in: {input_path}")
            return []
        print(f"Found {len(transcript_files)} transcript file(s) in {input_path}")
    else:
        print(f"Error: Invalid path: {input_path}")
        sys.exit(1)

    results = []
    for tf in transcript_files:
        if output and target.is_file() and not Path(output).is_dir():
            out_file = Path(output)
        else:
            out_dir = Path(output) if output else Path("Output")
            out_file = out_dir / f"{tf.stem}_summary.md"

        success = summarize_single_transcript(
            transcript_path=tf,
            output_path=out_file,
            language=language,
            provider=provider,
            openai_model=openai_model,
            anthropic_model=anthropic_model,
            gemini_model=gemini_model,
        )
        if success:
            results.append(out_file)

    return results


def main():
    parser = argparse.ArgumentParser(description="Summarize transcripts into markdown files.")
    parser.add_argument("transcript", help="Path to transcript file or directory of transcripts")
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown file path or directory (default: Output/<name>_summary.md)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Output language (default: auto-detect from .lang file or english)",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "ollama", "openai", "anthropic", "gemini"],
        help="LLM provider: auto (ollama -> anthropic -> gemini -> openai), ollama, openai, anthropic, or gemini",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model to use (default: gpt-4o-mini, e.g., gpt-4o, o3-mini)",
    )
    parser.add_argument(
        "--anthropic-model",
        default=DEFAULT_ANTHROPIC_MODEL,
        help="Anthropic model to use (default: claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model to use (default: gemini-2.5-flash)",
    )
    args = parser.parse_args()

    summarize_transcripts(
        input_path=args.transcript,
        output=args.output,
        language=args.language,
        provider=args.provider,
        openai_model=args.openai_model,
        anthropic_model=args.anthropic_model,
        gemini_model=args.gemini_model,
    )


if __name__ == "__main__":
    main()



