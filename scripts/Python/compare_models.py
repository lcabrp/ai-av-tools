import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.Python.summarize import (
    generate_with_anthropic,
    generate_with_gemini,
    generate_with_openai_compatible,
    OpenAI,
    OLLAMA_BASE_URL,
    os,
)


MODELS_TO_TEST = [
    {"name": "gemma4_12b", "provider": "ollama", "model": "gemma4:12b"},
    {"name": "gpt_luna", "provider": "openai", "model": "gpt-5.6-luna"},
    {"name": "gpt_terra", "provider": "openai", "model": "gpt-5.6-terra"},
    {"name": "gpt_sol", "provider": "openai", "model": "gpt-5.6-sol"},
    {"name": "claude_sonnet", "provider": "anthropic", "model": "claude-sonnet-4-6"},
    {"name": "gemini_flash", "provider": "gemini", "model": "gemini-flash-latest"},
]



def run_comparison(transcript_path: str, output_dir: str = "Output"):
    tf = Path(transcript_path)
    if not tf.exists():
        print(f"Error: File not found: {transcript_path}")
        sys.exit(1)

    transcript = tf.read_text(encoding="utf-8")
    lang_file = tf.with_suffix(".lang")
    language = lang_file.read_text(encoding="utf-8").strip() if lang_file.exists() else "spanish"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting comparison test for transcript: {tf.name}")
    print(f"Detected language: {language}\n" + "=" * 50)

    results = []

    for item in MODELS_TO_TEST:
        tag = item["name"]
        provider = item["provider"]
        model = item["model"]
        out_file = out_dir / f"{tf.stem}_summary_{tag}.md"

        print(f"\n[Running {tag}] Provider: {provider} | Model: {model}...")
        summary = None
        try:
            if provider == "ollama":
                client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
                summary = generate_with_openai_compatible(client, model, transcript, language)
            elif provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("  Skipping: OPENAI_API_KEY missing.")
                    continue
                client = OpenAI(api_key=api_key)
                summary = generate_with_openai_compatible(client, model, transcript, language)
            elif provider == "anthropic":
                summary = generate_with_anthropic(model, transcript, language)
            elif provider == "gemini":
                summary = generate_with_gemini(model, transcript, language)

            if summary:
                abs_out_file = out_file.resolve()
                abs_out_file.parent.mkdir(parents=True, exist_ok=True)
                abs_out_file.write_text(summary, encoding="utf-8")
                print(f"  SUCCESS! Saved to: {abs_out_file}")
                results.append({"tag": tag, "model": model, "file": abs_out_file})

        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n" + "=" * 50)
    print("Comparison runs completed! Summary files generated:")
    for r in results:
        print(f" - {r['tag']} ({r['model']}): {r['file']}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-model summary comparison test.")
    parser.add_argument(
        "transcript",
        default="Output/transcripts/bienvenido-agp.txt",
        nargs="?",
        help="Path to transcript file (default: Output/transcripts/bienvenido-agp.txt)",
    )
    parser.add_argument(
        "--output-dir",
        default="Output",
        help="Directory to save output summaries (default: Output)",
    )
    args = parser.parse_args()
    run_comparison(args.transcript, args.output_dir)


if __name__ == "__main__":
    main()
