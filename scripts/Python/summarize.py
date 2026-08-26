import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Explicit path avoids dotenv's stack-frame lookup, which is unreliable when
# this module is imported from different entry points (e.g. compare_models.py).
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_PRIMARY_LOCAL = "gemma4:12b"
DEFAULT_SECONDARY_LOCAL = "gemma4:e4b"
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")


def build_system_prompt(language: str) -> str:
    return (
        f"You are a meticulous assistant that produces detailed, action-ready markdown summaries of transcripts. "
        f"Your top priority is to NOT lose specific details — smaller models tend to over-compress, so err on the side "
        f"of including too much rather than too little.\n\n"
        f"Always preserve, verbatim where possible:\n"
        f"- Numbers: prices, percentages, dates, durations, quantities (e.g. CPM values, costs per country, timeframes).\n"
        f"- Proper nouns: people's names, tool/product/platform names, company names, URLs, emails.\n"
        f"- Step-by-step instructions or processes, in the exact order given, as numbered lists.\n"
        f"- Cause-and-effect or conditional statements (\"if X happens, do Y\").\n"
        f"- Concrete action items, decisions, deadlines, and next steps.\n\n"
        f"Structure the output as markdown with:\n"
        f"1. A title.\n"
        f"2. A short table of contents / key topics list.\n"
        f"3. One section per topic, using nested bullet points for sub-details (not just one flat bullet per topic).\n"
        f"4. A final 'Action Items' or 'Next Steps' section if the transcript mentions any.\n\n"
        f"Do not merge distinct details into vague generalizations. If unsure whether something is important, include it. "
        f"Write the entire summary in {language}."
    )


def build_user_prompt(transcript: str) -> str:
    return (
        f"Summarize the following transcript into a detailed markdown document. "
        f"Follow the system instructions strictly, especially about preserving specific numbers, names, and steps:\n\n{transcript}"
    )


def chunk_transcript(transcript: str, max_chars: int = 3000) -> list[str]:
    """Split transcript into chunks at sentence boundaries, each under max_chars.

    Small local models (e.g. on 4-8GB VRAM laptops) have limited effective
    context, so keeping chunks small and self-contained avoids the model
    running out of attention budget and silently dropping sections.
    """
    sentences = transcript.replace("\n", " ").split(". ")
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        piece = sentence if sentence.endswith(".") else sentence + "."
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current.strip())
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_chunk_extraction_prompt(chunk: str, index: int, total: int, language: str) -> str:
    return (
        f"This is fragment {index}/{total} of a longer transcript (do not summarize the whole "
        f"conversation, only this fragment). Extract ALL concrete details mentioned here as a flat "
        f"markdown bullet list: numbers, prices, dates, names, tool/platform names, steps, and "
        f"decisions. Do not omit anything specific, even if it seems minor. Do not add headers, "
        f"titles, or commentary — only bullet points. Write in {language}.\n\n"
        f"Fragment:\n{chunk}"
    )


def build_reduce_prompt(combined_notes: str, language: str) -> str:
    return (
        f"Below are detailed notes extracted from consecutive fragments of the same transcript, in order. "
        f"Merge them into a single, well-organized markdown summary. Do not drop any specific detail "
        f"(numbers, names, steps, decisions) already present in the notes — only reorganize, deduplicate, "
        f"and structure them. Follow the system instructions for structure and write in {language}.\n\n"
        f"Notes:\n{combined_notes}"
    )


def generate_chunked(
    call_fn, transcript: str, language: str, chunk_chars: int = 3000
) -> str:
    """Provider-agnostic map-reduce summarization: summarize each chunk separately via
    call_fn(system_prompt, user_prompt) -> str, then merge into one final summary."""
    chunks = chunk_transcript(transcript, max_chars=chunk_chars)
    print(f"  Chunked transcript into {len(chunks)} piece(s) of ~{chunk_chars} chars...")

    chunk_notes = []
    for i, chunk in enumerate(chunks, start=1):
        note = call_fn(
            "You extract specific facts from transcript fragments precisely.",
            build_chunk_extraction_prompt(chunk, i, len(chunks), language),
        )
        chunk_notes.append(note)
        print(f"    Processed chunk {i}/{len(chunks)}")

    combined_notes = "\n\n".join(chunk_notes)
    return call_fn(build_system_prompt(language), build_reduce_prompt(combined_notes, language))


def generate_with_ollama_chunked(
    client: OpenAI, model: str, transcript: str, language: str, chunk_chars: int = 3000
) -> str:
    """Map-reduce summarization for local models: summarize each chunk separately,
    then merge, so long transcripts don't exceed a small model's effective context."""

    def call_fn(system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    return generate_chunked(call_fn, transcript, language, chunk_chars)


def generate_with_gemini_chunked(model: str, transcript: str, language: str, chunk_chars: int = 15000) -> str:
    """Map-reduce summarization for Gemini. Gemini's context window is large, so this uses
    much bigger chunks than local models — just a few calls to double-check nothing near
    chunk boundaries gets compressed, not to work around a small context window."""
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY / GOOGLE_API_KEY not found in environment or .env file.")
    client = genai.Client(api_key=api_key)

    def call_fn(system_prompt: str, user_prompt: str) -> str:
        response = client.models.generate_content(
            model=model,
            contents=f"{system_prompt}\n\n{user_prompt}",
        )
        return response.text

    return generate_chunked(call_fn, transcript, language, chunk_chars)



def generate_with_openai_compatible(client: OpenAI, model: str, transcript: str, language: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": build_system_prompt(language),
            },
            {
                "role": "user",
                "content": build_user_prompt(transcript),
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
        system=build_system_prompt(language),
        messages=[
            {
                "role": "user",
                "content": build_user_prompt(transcript),
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
    prompt = f"{build_system_prompt(language)}\n\n{build_user_prompt(transcript)}"
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
    use_chunking: bool = False,
    chunk_chars: int = 3000,
    gemini_chunk_chars: int = 15000,
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

    # gemini-flash-latest is the default: best detail retention in benchmarks. Long
    # transcripts are automatically chunked (map-reduce) since that measurably improved
    # detail retention (e.g. captured individually named cases a single-shot call missed).
    if provider in ("auto", "gemini"):
        try:
            if len(transcript) > gemini_chunk_chars:
                print(f"Generating summary with Gemini ({gemini_model}), chunked (long transcript)...")
                summary = generate_with_gemini_chunked(gemini_model, transcript, language, gemini_chunk_chars)
            else:
                print(f"Generating summary with Gemini ({gemini_model})...")
                summary = generate_with_gemini(gemini_model, transcript, language)
        except Exception as e:
            print(f"Gemini API failed: {e}")

    if summary is None and provider in ("auto", "anthropic"):
        try:
            print(f"Generating summary with Anthropic ({anthropic_model})...")
            summary = generate_with_anthropic(anthropic_model, transcript, language)
        except Exception as e:
            print(f"Anthropic API failed: {e}")

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

    if summary is None and provider in ("auto", "ollama"):
        ollama_client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
        )
        generate_local = (
            (lambda m: generate_with_ollama_chunked(ollama_client, m, transcript, language, chunk_chars))
            if use_chunking
            else (lambda m: generate_with_openai_compatible(ollama_client, m, transcript, language))
        )
        try:
            print(f"Generating summary with local Ollama ({primary_local_model}) for {transcript_path.name}...")
            summary = generate_local(primary_local_model)
        except Exception as e:
            print(f"Ollama primary model ({primary_local_model}) failed: {e}")
            if provider == "auto":
                try:
                    print(f"Falling back to secondary local Ollama model ({secondary_local_model})...")
                    summary = generate_local(secondary_local_model)
                except Exception as e2:
                    print(f"Ollama secondary model ({secondary_local_model}) failed: {e2}")

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
    use_chunking: bool = False,
    chunk_chars: int = 3000,
    gemini_chunk_chars: int = 15000,
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
            use_chunking=use_chunking,
            chunk_chars=chunk_chars,
            gemini_chunk_chars=gemini_chunk_chars,
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
        help="LLM provider: auto (gemini -> anthropic -> openai -> ollama), ollama, openai, anthropic, or gemini",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model to use (default: gpt-5.6-luna, e.g., gpt-5.6-terra, gpt-5.6-sol)",
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

    summarize_transcripts(
        input_path=args.transcript,
        output=args.output,
        language=args.language,
        provider=args.provider,
        openai_model=args.openai_model,
        anthropic_model=args.anthropic_model,
        gemini_model=args.gemini_model,
        use_chunking=args.chunked,
        chunk_chars=args.chunk_chars,
        gemini_chunk_chars=args.gemini_chunk_chars,
    )


if __name__ == "__main__":
    main()



