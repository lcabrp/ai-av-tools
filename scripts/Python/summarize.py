import argparse
import json
import os
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

try:
    from scripts.Python.ops_log import log_operation
except ImportError:  # run directly as a script: siblings are top-level on sys.path
    from ops_log import log_operation

# Windows consoles default to cp1252, which crashes on emoji/CJK in file
# names (same fix as benchmark_transcription.py).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Explicit path avoids dotenv's stack-frame lookup, which is unreliable when
# this module is imported from different entry points (e.g. compare_models.py).
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_PRIMARY_LOCAL = "gemma4:12b"
DEFAULT_SECONDARY_LOCAL = "gemma4:e4b"
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
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
        # Whisper output can contain very long stretches without periods.
        # Hard-wrap those stretches so no chunk exceeds the model's budget.
        for part in textwrap.wrap(piece, width=max_chars, break_long_words=True,
                                  break_on_hyphens=False) or [piece]:
            if current and len(current) + len(part) + 1 > max_chars:
                chunks.append(current.strip())
                current = part
            else:
                current = f"{current} {part}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_chunk_extraction_prompt(chunk: str, index: int, total: int, language: str) -> str:
    return (
        f"This is fragment {index}/{total} of a longer transcript (do not summarize the whole "
        f"conversation, only this fragment). Extract ALL concrete details as a flat markdown bullet "
        f"list. Preserve who said or asked each item and label it as [Question], [Answer], [Example], "
        f"[Decision], [Action: owner], [Correction], or [Unresolved] when the category is clear. "
        f"Keep numbers, prices, dates, names, tools, URLs, ordered steps, deadlines, conditions, and "
        f"speaker uncertainty verbatim where possible. Explicitly preserve rejected ideas and later "
        f"corrections; never turn a participant's question, assumption, or numerical example into a "
        f"confirmed fact or recommendation. Do not infer missing facts or normalize uncertain names. "
        f"Do not omit anything specific, even if it seems minor. Do not add headers, titles, or "
        f"commentary — only bullets. Write in {language}.\n\n"
        f"Fragment:\n{chunk}"
    )


def build_reduce_prompt(combined_notes: str, language: str) -> str:
    return (
        f"Below are detailed notes extracted from consecutive transcript fragments, in order. Merge "
        f"them without changing epistemic status: questions remain questions, examples remain examples, "
        f"rejected ideas remain rejected, and only explicit answers or decisions become recommendations. "
        f"Preserve speaker/owner attribution, uncertain names, corrections, contradictions, numbers, names, "
        f"steps, deadlines, conditions, unresolved items, and individual cases. Deduplicate only genuinely "
        f"repeated information. Do not resolve contradictions yourself or add outside facts. Produce a "
        f"well-organized markdown summary following the system instructions in {language}.\n\n"
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


def group_notes(notes: list[str], max_chars: int) -> list[str]:
    groups = []
    current = []
    current_chars = 0
    for note in notes:
        added_chars = len(note) + (2 if current else 0)
        if current and current_chars + added_chars > max_chars:
            groups.append("\n\n".join(current))
            current = [note]
            current_chars = len(note)
        else:
            current.append(note)
            current_chars += added_chars
    if current:
        groups.append("\n\n".join(current))
    return groups


def generate_hierarchical_chunked(
    extract_fn, reduce_fn, transcript: str, language: str,
    chunk_chars: int = 3000, reduce_chars: int = 12000,
) -> str:
    chunks = chunk_transcript(transcript, max_chars=chunk_chars)
    print(f"  Chunked transcript into {len(chunks)} piece(s) of ~{chunk_chars} chars...", flush=True)

    notes = []
    for i, chunk in enumerate(chunks, start=1):
        notes.append(extract_fn(
            "Extract transcript facts precisely. Preserve attribution and epistemic status.",
            build_chunk_extraction_prompt(chunk, i, len(chunks), language),
        ))
        print(f"    Extracted chunk {i}/{len(chunks)}", flush=True)

    level = 1
    while len(notes) > 1:
        groups = group_notes(notes, reduce_chars)
        if len(groups) == 1:
            break
        reduced = []
        for i, group in enumerate(groups, start=1):
            reduced.append(reduce_fn(
                "Consolidate transcript notes without losing attribution, corrections, or concrete facts.",
                build_reduce_prompt(group, language),
            ))
            print(f"    Reduced level {level} group {i}/{len(groups)}", flush=True)
        notes = reduced
        level += 1

    combined_notes = "\n\n".join(notes)
    return reduce_fn(build_system_prompt(language), build_reduce_prompt(combined_notes, language))


def generate_with_ollama(
    model: str, system_prompt: str, user_prompt: str,
    context_tokens: int = 32768, temperature: float = 0.1,
    num_predict: int = 8192,
) -> str:
    base_url = OLLAMA_BASE_URL.removesuffix("/v1").rstrip("/")
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "num_ctx": context_tokens,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    return result["message"]["content"]


def generate_with_ollama_chunked(
    model: str, transcript: str, language: str, chunk_chars: int = 3000,
    reduce_chars: int = 12000, context_tokens: int = 32768,
    temperature: float = 0.1,
) -> str:
    """Map-reduce summarization for local models: summarize each chunk separately,
    then merge, so long transcripts don't exceed a small model's effective context."""

    def extract_fn(system_prompt: str, user_prompt: str) -> str:
        return generate_with_ollama(
            model, system_prompt, user_prompt, context_tokens, temperature, 1024,
        )

    def reduce_fn(system_prompt: str, user_prompt: str) -> str:
        return generate_with_ollama(
            model, system_prompt, user_prompt, context_tokens, temperature, 4096,
        )

    return generate_hierarchical_chunked(
        extract_fn, reduce_fn, transcript, language, chunk_chars, reduce_chars,
    )


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


def provider_tag(provider: str, model: str | None = None) -> str:
    """Short filename tag identifying the provider (and model variant for OpenAI).

    Examples: gemini, anthropic, openai_luna, openai_terra, ollama.
    Keeps per-provider summaries side-by-side instead of overwriting each other.
    """
    if provider == "openai" and model:
        # gpt-6-luna / gpt-5.6-luna -> openai_luna (drop the generic gpt prefix)
        variant = model.split("-")[-1]
        return f"openai_{variant}"
    return provider


def summarize_single_transcript(
    transcript_path: Path,
    output_path: Path,
    language: str | None = None,
    provider: str = "auto",
    openai_model: str = DEFAULT_OPENAI_MODEL,
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    use_chunking: bool = False,
    chunk_chars: int = 3000,
    gemini_chunk_chars: int = 15000,
    local_model: str = DEFAULT_PRIMARY_LOCAL,
    local_fallback_model: str = DEFAULT_SECONDARY_LOCAL,
    local_context: int = 32768,
    local_temperature: float = 0.1,
    local_reduce_chars: int = 40000,
) -> bool:
    if not transcript_path.exists():
        print(f"Error: Transcript file not found: {transcript_path}")
        return False

    transcript = transcript_path.read_text(encoding="utf-8")
    if not transcript.strip():
        print(f"Warning: Transcript {transcript_path.name} is empty. Skipping.")
        return False

    # Tag the default output name with the provider so different providers'
    # summaries coexist instead of overwriting each other. Explicit --output
    # file paths are used as-is. "auto" resolves to whichever provider succeeds.
    if output_path.name == f"{transcript_path.stem}_summary.md":
        def _tag_output(resolved_provider: str, resolved_model: str | None) -> None:
            nonlocal output_path
            output_path = output_path.with_name(
                f"{transcript_path.stem}_summary_{provider_tag(resolved_provider, resolved_model)}.md"
            )
    else:
        _tag_output = lambda rp, rm: None

    if not language:
        lang_file = transcript_path.with_suffix(".lang")
        if lang_file.exists():
            language = lang_file.read_text(encoding="utf-8").strip()
        else:
            language = "english"

    summary = None
    used_config = ""
    t_start = time.perf_counter()

    # gemini-flash-latest is the default: best detail retention in benchmarks. Long
    # transcripts are automatically chunked (map-reduce) since that measurably improved
    # detail retention (e.g. captured individually named cases a single-shot call missed).
    if provider in ("auto", "gemini"):
        try:
            if len(transcript) > gemini_chunk_chars:
                print(f"Generating summary with Gemini ({gemini_model}), chunked (long transcript)...")
                summary = generate_with_gemini_chunked(gemini_model, transcript, language, gemini_chunk_chars)
                used_config = f"gemini/{gemini_model} chunked({gemini_chunk_chars})"
            else:
                print(f"Generating summary with Gemini ({gemini_model})...")
                summary = generate_with_gemini(gemini_model, transcript, language)
                used_config = f"gemini/{gemini_model} single-shot"
            _tag_output("gemini", gemini_model)
        except Exception as e:
            print(f"Gemini API failed: {e}")

    if summary is None and provider in ("auto", "anthropic"):
        try:
            print(f"Generating summary with Anthropic ({anthropic_model})...")
            summary = generate_with_anthropic(anthropic_model, transcript, language)
            used_config = f"anthropic/{anthropic_model}"
            _tag_output("anthropic", anthropic_model)
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
                used_config = f"openai/{openai_model}"
                _tag_output("openai", openai_model)
            except Exception as e:
                print(f"OpenAI API failed: {e}")

    if summary is None and provider in ("auto", "ollama"):
        generate_local = (
            (lambda m: generate_with_ollama_chunked(
                m, transcript, language, chunk_chars,
                local_reduce_chars, local_context, local_temperature,
            ))
            if use_chunking
            else (lambda m: generate_with_ollama(
                m, build_system_prompt(language), build_user_prompt(transcript),
                local_context, local_temperature,
            ))
        )
        mode = f"chunked({chunk_chars})" if use_chunking else "single-shot"
        try:
            print(f"Generating summary with local Ollama ({local_model}) for {transcript_path.name}...")
            summary = generate_local(local_model)
            used_config = f"ollama/{local_model} {mode}"
            _tag_output("ollama", local_model)
        except Exception as e:
            print(f"Ollama primary model ({local_model}) failed: {e}")
            if provider == "auto":
                try:
                    print(f"Falling back to secondary local Ollama model ({local_fallback_model})...")
                    summary = generate_local(local_fallback_model)
                    used_config = f"ollama/{local_fallback_model} {mode} (fallback)"
                    _tag_output("ollama", local_fallback_model)
                except Exception as e2:
                    print(f"Ollama secondary model ({local_fallback_model}) failed: {e2}")

    elapsed = time.perf_counter() - t_start

    if summary is None:
        print(f"Error: All summary providers failed for {transcript_path.name}.")
        log_operation(
            operation="summarize",
            input_file=transcript_path.name,
            config=f"{provider} (all failed)",
            elapsed_seconds=elapsed,
            output_file="",
            input_chars=len(transcript),
            output_chars=None,
            status="failed",
        )
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    print(f"Summary saved to: {output_path}")
    log_operation(
        operation="summarize",
        input_file=transcript_path.name,
        config=used_config or provider,
        elapsed_seconds=elapsed,
        output_file=output_path.name,
        input_chars=len(transcript),
        output_chars=len(summary),
        status="ok",
    )
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
    local_model: str = DEFAULT_PRIMARY_LOCAL,
    local_fallback_model: str = DEFAULT_SECONDARY_LOCAL,
    local_context: int = 32768,
    local_temperature: float = 0.1,
    local_reduce_chars: int = 40000,
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
            local_model=local_model,
            local_fallback_model=local_fallback_model,
            local_context=local_context,
            local_temperature=local_temperature,
            local_reduce_chars=local_reduce_chars,
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
        help="Output markdown file path or directory (default: Output/<name>_summary_<provider>.md)",
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
        help="OpenAI model to use (default: gpt-6-luna, e.g., gpt-6-sol, gpt-6-astra, gpt-5.6-terra)",
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
        "--local-model",
        default=DEFAULT_PRIMARY_LOCAL,
        help=f"Ollama model to use (default: {DEFAULT_PRIMARY_LOCAL})",
    )
    parser.add_argument(
        "--local-fallback-model",
        default=DEFAULT_SECONDARY_LOCAL,
        help=f"Ollama fallback model used by auto mode (default: {DEFAULT_SECONDARY_LOCAL})",
    )
    parser.add_argument(
        "--local-context",
        type=int,
        default=32768,
        help="Ollama context window in tokens (default: 32768)",
    )
    parser.add_argument(
        "--local-temperature",
        type=float,
        default=0.1,
        help="Ollama sampling temperature (default: 0.1)",
    )
    parser.add_argument(
        "--local-reduce-chars",
        type=int,
        default=40000,
        help="Maximum note characters per hierarchical reduction group (default: 40000)",
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
        local_model=args.local_model,
        local_fallback_model=args.local_fallback_model,
        local_context=args.local_context,
        local_temperature=args.local_temperature,
        local_reduce_chars=args.local_reduce_chars,
    )


if __name__ == "__main__":
    main()



