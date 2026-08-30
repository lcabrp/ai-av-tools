# Whisper Transcription Benchmark: openai-whisper vs faster-whisper

Why this exists: `scripts/Python/transcribe.py` uses openai-whisper `medium` because `base` was
not accurate enough for real conversations. `faster-whisper` claims ~4x speedups but initially
appeared no faster here — that turned out to be a flawed test (see "The CPU trap" below).

Script: `scripts/Python/benchmark_transcription.py` (run with `uv run`).

## Reference results — NVIDIA T1000 8GB

System: T1000 8GB (Turing TU117, compute capability 7.5, **no tensor cores**), driver 595.79,
torch 2.6.0+cu124, Python 3.13, openai-whisper 20250625, faster-whisper 1.2.1.
Audio: `TaxPreparer_KTA_Institute_2026-08-26.mp3` (53.4 min Spanish conversation).

60s slice:

| Backend | Config | Transcribe | Notes |
|---|---|---|---|
| openai-whisper | tiny / base / small / medium | 2.2s / 3.5s / 8.2s / 27.4s | quality climbs up to medium |
| faster-whisper medium | float16 | 27.4s | slow on this GPU |
| faster-whisper medium | float32 | 11.5s | |
| faster-whisper medium | int8 / int8_float16 | **9.0s** | fastest, identical quality |

300s slice:

| Backend | Config | Transcribe | Chars |
|---|---|---|---|
| openai-whisper medium | fp16 | 109.2s | 3556 |
| faster-whisper medium | **int8_float16** | **40.9s** | 3778 |
| faster-whisper medium | float16 | 119.9s | 3743 |

Full 53.4-min file with `faster-whisper medium cuda int8_float16 beam_size=5`:
**442.8s (7.4 min)**, 457 segments, 41,296 chars
(transcript kept in `bench_output/faster_medium_int8_full.txt`).

## Findings (T1000)

1. **Winner: `faster-whisper` + `compute_type="int8_float16"`** — ~2.7x faster than
   openai-whisper `medium` fp16, with equal-or-better text (more characters, better
   punctuation). No visible quality loss vs fp32.
2. **`float16` is the *slowest* option on this GPU.** The TU117 has no tensor cores; the fast
   path is INT8 (DP4A). On GPUs *with* tensor cores (RTX 20/30/40/50, RTX A-series) `float16`
   will likely beat or tie `int8_float16` — benchmark both.
3. `medium` remains the quality floor (`base` insufficient for this content), so speedups must
   come from the runtime, not a smaller model.
4. Expect roughly linear scaling: 300s slice times ÷ 300 × file duration ≈ full-file time.

## The CPU trap (what went wrong initially)

The first "benchmark" used bare `pip install` + `python` instead of `uv`. That installed
CPU-only torch 2.13 into the *global* Python environment, and everything silently ran on CPU:

- openai-whisper warned `FP16 is not supported on CPU; using FP32 instead` and ran ~3x slower;
- faster-whisper crashed with `Library cublas64_12.dll is not found` (ctranslate2 could not
  find a usable CUDA runtime), which looked like a GPU driver problem but wasn't.

Rules to avoid this:

- In this repo, always run through `uv run` (or the project `.venv`), never bare `python`/`pip`.
- Verify before benchmarking: `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
  must print `...+cu124 True <GPU name>`.
- The benchmark script prints a System header and warns if CUDA is unavailable; trust that header.
- All packages that were installed into the global environment with pip during the failed tests
  (openai-whisper, faster-whisper, CPU torch 2.13 + ~30 deps) have been uninstalled.

## How to reproduce on a different PC

Prerequisites: NVIDIA GPU with a recent driver, [uv](https://docs.astral.sh/uv/), and ffmpeg
(`winget install --id Gyan.FFmpeg`).

```bash
git clone <this repo> && cd ai-av-tools
uv sync                # installs torch from the cu124 index + whisper + faster-whisper

# 1) Verify GPU is visible to torch (must say True and show the GPU name)
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2) Quick benchmark: first 300s of the file (same settings as the T1000 reference run)
uv run scripts/Python/benchmark_transcription.py "E:\path\to\TaxPreparer_KTA_Institute_2026-08-26.mp3" --slice 300 --language es

# 3) Full-file benchmark (takes a while)
uv run scripts/Python/benchmark_transcription.py "E:\path\to\file.mp3" --slice 0 --language es

# Optional: also time tiny/base/small/medium on openai-whisper
uv run scripts/Python/benchmark_transcription.py file.mp3 --all-models
```

The script saves each transcript plus `bench_output/benchmark_report.txt` (system header +
timing table). When comparing GPUs, share that report file; note especially:

- GPU model, VRAM, compute capability, driver version
- which `compute_type` won: try `float16` vs `int8_float16` — tensor-core GPUs usually prefer
  `float16`, non-tensor-core Turing (T1000/GTX 16-series) prefers `int8_float16`
- whether anything fell back to CPU (chars/time vs the T1000 reference above is a good sanity check)

## Follow-up

Consider switching `scripts/Python/transcribe.py` from openai-whisper to
`faster-whisper` (`device="cuda"`, `compute_type="int8_float16"` on the T1000) for ~2.7x
throughput at the same quality. Summarization stays on Gemini (best detail retention).
