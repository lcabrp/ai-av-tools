"""Benchmark local Ollama models with the production summarization pipeline."""

import argparse
import csv
import json
import re
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.Python.summarize import (
    generate_with_ollama_chunked,
)


FIELDS = [
    "recorded_at", "machine", "gpu", "model", "parameter_size", "quantization",
    "transcript", "transcript_chars", "language", "chunk_chars", "reduce_chars",
    "context_tokens", "temperature", "elapsed_seconds", "summary_chars",
    "summary_words", "summary_lines", "status", "output_file",
]
KEY_FIELDS = ["machine", "model", "transcript", "chunk_chars", "reduce_chars", "context_tokens"]


def ollama_show(model: str) -> dict:
    request = urllib.request.Request(
        "http://localhost:11434/api/show",
        data=json.dumps({"model": model}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def safe_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model)


def save_results(path: Path, new_records: list[dict]):
    records = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"Unexpected benchmark CSV columns in {path}")
            records = list(reader)

    new_keys = {tuple(record[field] for field in KEY_FIELDS) for record in new_records}
    records = [
        record for record in records
        if tuple(record[field] for field in KEY_FIELDS) not in new_keys
    ]
    records.extend(new_records)
    records.sort(key=lambda record: tuple(record[field] for field in KEY_FIELDS))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description="Benchmark local Ollama transcript summaries.")
    parser.add_argument("transcript", help="Transcript text file")
    parser.add_argument("--models", nargs="+", required=True, help="Ollama model tags to test")
    parser.add_argument("--output-dir", default="Output/local-summary-benchmarks")
    parser.add_argument("--results-csv", default="Docs/local-summary-benchmark-results.csv")
    parser.add_argument("--language", default=None)
    parser.add_argument("--chunk-chars", type=int, default=3000)
    parser.add_argument("--reduce-chars", type=int, default=40000)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        raise SystemExit(f"Transcript not found: {transcript_path}")
    transcript = transcript_path.read_text(encoding="utf-8")
    language = args.language
    if not language and transcript_path.with_suffix(".lang").exists():
        language = transcript_path.with_suffix(".lang").read_text(encoding="utf-8").strip()
    language = language or "english"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recorded_at = datetime.now().isoformat(timespec="seconds")
    machine = socket.gethostname()
    gpu = gpu_name()
    records = []

    for model in args.models:
        output_file = output_dir / f"{transcript_path.stem}_{safe_model_name(model)}.md"
        print(f"\n=== {model} ===", flush=True)
        started = time.perf_counter()
        status = "ok"
        summary = ""
        try:
            model_info = ollama_show(model)
            details = model_info.get("details", {})
            summary = generate_with_ollama_chunked(
                model, transcript, language, args.chunk_chars,
                args.reduce_chars, args.context, args.temperature,
            )
            if len(transcript) > 10000 and len(summary.strip()) < 1000:
                raise ValueError(
                    f"Summary is unexpectedly short ({len(summary.strip())} characters)"
                )
            output_file.write_text(summary, encoding="utf-8")
        except Exception as error:
            details = {}
            status = f"{type(error).__name__}: {error}"
            print(f"FAILED: {status}", flush=True)
        elapsed = time.perf_counter() - started
        records.append({
            "recorded_at": recorded_at,
            "machine": machine,
            "gpu": gpu,
            "model": model,
            "parameter_size": details.get("parameter_size", ""),
            "quantization": details.get("quantization_level", ""),
            "transcript": transcript_path.name,
            "transcript_chars": str(len(transcript)),
            "language": language,
            "chunk_chars": str(args.chunk_chars),
            "reduce_chars": str(args.reduce_chars),
            "context_tokens": str(args.context),
            "temperature": str(args.temperature),
            "elapsed_seconds": f"{elapsed:.1f}",
            "summary_chars": str(len(summary)),
            "summary_words": str(len(summary.split())),
            "summary_lines": str(len(summary.splitlines())),
            "status": status,
            "output_file": str(output_file),
        })
        save_results(Path(args.results_csv), records)
        print(f"Completed in {elapsed:.1f}s: {output_file}", flush=True)


if __name__ == "__main__":
    main()
