"""Benchmark transcription backends (openai-whisper vs faster-whisper) on this machine.

Prints a system-info header plus load/transcribe timings for each backend config and
saves every transcript (plus a report file) so outputs can be compared for quality.
See Docs/whisper-benchmark.md for recorded results and how to reproduce on another PC.
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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


def pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def system_header() -> list[str]:
    lines = [
        f"date: {datetime.now().isoformat(timespec='seconds')}",
        f"hostname: {socket.gethostname()}",
        f"python: {sys.version.split()[0]}",
        f"torch: {torch.__version__} (cuda_available={torch.cuda.is_available()}, cuda_runtime={torch.version.cuda})",
        f"openai-whisper: {pkg_version('openai-whisper')}",
        f"faster-whisper: {pkg_version('faster-whisper')}",
    ]
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        lines.append(
            f"gpu: {p.name} (compute capability {p.major}.{p.minor}, "
            f"{p.total_memory / 1024**3:.1f} GB VRAM, {p.multi_processor_count} SMs)"
        )
    else:
        lines.append("gpu: none visible to torch -- CPU-only run")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        lines.append(f"nvidia driver: {out.stdout.strip().splitlines()[0]}")
    except Exception:
        lines.append("nvidia driver: nvidia-smi not found")
    return lines


def make_slice(audio: Path, seconds: int, outdir: Path) -> Path:
    sliced = outdir / f"{audio.stem}_first{seconds}s.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "0", "-t", str(seconds), "-i", str(audio), str(sliced)],
        check=True, capture_output=True,
    )
    return sliced


def run_openai_whisper(audio: Path, model_size: str, language: str | None, outdir: Path) -> dict:
    row = {"backend": "openai-whisper", "config": model_size, "device": "",
           "load_s": None, "run_s": None, "total_s": None, "chars": None,
           "output": "", "error": None}
    try:
        import whisper
        fp16 = torch.cuda.is_available()
        compute = "fp16" if fp16 else "fp32"
        t0 = time.perf_counter()
        model = whisper.load_model(model_size)
        t1 = time.perf_counter()
        result = model.transcribe(str(audio), language=language, fp16=fp16)
        t2 = time.perf_counter()
        text = result["text"].strip()
        row["device"] = str(next(model.parameters()).device)
        row["config"] = f"{model_size} {compute}"
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        out_file = outdir / f"openai-whisper-{model_size}-{compute}.txt"
        out_file.write_text(text, encoding="utf-8")
        row.update(load_s=t1 - t0, run_s=t2 - t1, total_s=t2 - t0,
                   chars=len(text), output=str(out_file))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def run_faster_whisper(audio: Path, model_size: str, compute_type: str,
                       language: str | None, outdir: Path) -> dict:
    row = {"backend": "faster-whisper", "config": f"{model_size} {compute_type}",
           "device": "cuda" if torch.cuda.is_available() else "cpu",
           "load_s": None, "run_s": None, "total_s": None, "chars": None,
           "output": "", "error": None}
    try:
        from faster_whisper import WhisperModel
        t0 = time.perf_counter()
        model = WhisperModel(model_size, device=row["device"], compute_type=compute_type)
        t1 = time.perf_counter()
        segments, info = model.transcribe(str(audio), language=language, beam_size=5)
        text = " ".join(seg.text for seg in segments).strip()
        t2 = time.perf_counter()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        out_file = outdir / f"faster-whisper-{model_size}-{compute_type}.txt"
        out_file.write_text(text, encoding="utf-8")
        row.update(load_s=t1 - t0, run_s=t2 - t1, total_s=t2 - t0,
                   chars=len(text), output=str(out_file))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def fmt(value, suffix=""):
    return f"{value:.1f}{suffix}" if isinstance(value, (int, float)) else "-"


def print_table(rows: list[dict]):
    header = f"{'backend':<15} {'config':<22} {'device':<6} {'load':>7} {'transcribe':>10} {'total':>7} {'chars':>7}  status"
    print(f"\n{header}")
    print("-" * len(header))
    for r in rows:
        status = "ok" if r["error"] is None else f"FAILED: {r['error'][:60]}"
        print(
            f"{r['backend']:<15} {r['config']:<22} {r['device']:<6} "
            f"{fmt(r['load_s']):>7} {fmt(r['run_s']):>10} {fmt(r['total_s']):>7} "
            f"{r['chars'] if r['chars'] is not None else '-':>7}  {status}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark openai-whisper vs faster-whisper on this machine.")
    parser.add_argument("audio", help="Audio/video file to benchmark")
    parser.add_argument("--slice", type=int, default=300, metavar="SECONDS",
                        help="Only benchmark the first N seconds (0 = whole file, default 300)")
    parser.add_argument("--language", default=None,
                        help="Force language code (e.g. 'es'); default is auto-detect")
    parser.add_argument("--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--all-models", action="store_true",
                        help="Also benchmark tiny/base/small/medium with openai-whisper")
    parser.add_argument("--skip-openai", action="store_true")
    parser.add_argument("--skip-faster", action="store_true")
    parser.add_argument("--output-dir", default="bench_output",
                        help="Where to save transcripts and the report (default: bench_output)")
    args = parser.parse_args()

    ensure_ffmpeg_in_path()

    audio = Path(args.audio).expanduser().resolve()
    if not audio.exists():
        sys.exit(f"Error: audio file not found: {audio}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.slice > 0:
        print(f"Extracting first {args.slice}s with ffmpeg ...")
        audio = make_slice(audio, args.slice, outdir)

    header = system_header()
    print("\n=== System ===")
    for line in header:
        print(line)
    if not torch.cuda.is_available():
        print("\nWARNING: torch reports CUDA is NOT available. Results will be CPU-based.\n"
              "If this machine has an NVIDIA GPU, the environment is wrong (see\n"
              "Docs/whisper-benchmark.md, 'The CPU trap').")

    print(f"\n=== Benchmarking: {audio.name} ===")
    rows: list[dict] = []

    if not args.skip_openai:
        sizes = ["tiny", "base", "small", "medium"] if args.all_models else [args.model]
        for size in sizes:
            print(f"[openai-whisper {size}] running ...", flush=True)
            row = run_openai_whisper(audio, size, args.language, outdir)
            rows.append(row)
            print(f"[openai-whisper {row['config']}] transcribe={fmt(row['run_s'], 's')} "
                  f"chars={row['chars']} device={row['device']}" if not row["error"]
                  else f"[openai-whisper {size}] FAILED: {row['error']}", flush=True)

    if not args.skip_faster:
        compute_types = ["float16", "int8_float16", "float32"] if torch.cuda.is_available() else ["int8"]
        for compute in compute_types:
            print(f"[faster-whisper {args.model} {compute}] running ...", flush=True)
            row = run_faster_whisper(audio, args.model, compute, args.language, outdir)
            rows.append(row)
            print(f"[faster-whisper {row['config']}] transcribe={fmt(row['run_s'], 's')} "
                  f"chars={row['chars']}" if not row["error"]
                  else f"[faster-whisper {compute}] FAILED: {row['error']}", flush=True)

    print_table(rows)

    report_lines = header + ["", f"audio: {audio}", ""] + ["Results"]
    report_lines += [
        f"{r['backend']} | {r['config']} | device={r['device']} | load={fmt(r['load_s'], 's')} | "
        f"transcribe={fmt(r['run_s'], 's')} | total={fmt(r['total_s'], 's')} | chars={r['chars']} | "
        f"{r['output'] or r['error']}"
        for r in rows
    ]
    report = outdir / "benchmark_report.txt"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nTranscripts and report saved to: {outdir}/")
    print("When comparing GPUs, share bench_output/benchmark_report.txt")


if __name__ == "__main__":
    main()
