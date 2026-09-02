# Whisper Transcription Benchmark: openai-whisper vs faster-whisper

Why this exists: `scripts/Python/transcribe.py` uses openai-whisper `medium` because `base` was
not accurate enough for real conversations. `faster-whisper` claims ~4x speedups but initially
appeared no faster here — that turned out to be a flawed test (see "The CPU trap" below).

Script: `scripts/Python/benchmark_transcription.py` (run with `uv run`).

## Recommendation

Use **`faster-whisper` with the `medium` model** for local transcription. It runs the same
Whisper model through CTranslate2 and was substantially faster than `openai-whisper` without an
observed quality regression in these tests.

Choose the compute type for the GPU:

- Modern NVIDIA RTX GPUs with tensor cores: use `compute_type="float16"`. On the RTX 4070 it was
  effectively tied with INT8/FP16, so FP16 is the conservative default when quality is the priority.
- NVIDIA T1000 8GB and similar non-tensor-core Turing GPUs: use
  `compute_type="int8_float16"`. The T1000's INT8 path was about 2.9x faster than its FP16 path.
- Keep `medium` as the default model. Earlier testing found `base` too inaccurate for real
  conversations with accents, mumbled speech, and crosstalk.

`faster-whisper` also decodes media through PyAV, while `openai-whisper` shells out to FFmpeg.
FFmpeg is still required by the benchmark script to create a consistent timed slice.

The production transcription script still uses `openai-whisper`; migrating it to
`faster-whisper` is a separate implementation change.

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

## RTX 4070 comparison

System: XPS-8960, RTX 4070 12GB (Ada, compute capability 8.9), driver 595.95,
torch 2.6.0+cu124, Python 3.13.12, openai-whisper 20250625, faster-whisper 1.2.1.
The same first 300 seconds of the same Spanish recording were used.

| Backend | Config | T1000 | RTX 4070 | RTX speedup |
|---|---|---:|---:|---:|
| openai-whisper medium | fp16 | 109.2s | 20.9s | 5.2x |
| faster-whisper medium | float16 | 119.9s | 10.9s | 11.0x |
| faster-whisper medium | int8_float16 | 40.9s | 10.6s | 3.9x |

On the RTX 4070, faster-whisper FP16 took 10.9s versus 20.9s for openai-whisper FP16, a
1.9x backend speedup. INT8/FP16 was only 0.3s faster than FP16, which is too small to justify
quantization when the goal is speed without sacrificing quality. The FP16 transcript retained
3,794 characters and had 98.9% normalized word-sequence similarity to the FP32 transcript.

The 53.4-minute source should take roughly two minutes on the RTX 4070 if throughput scales
linearly, versus the measured 7.4 minutes on the T1000.

These quality comparisons are indicators, not word-error-rate measurements: there is no manually
verified reference transcript. Character count alone is not a quality score. The strongest result
is that FP16, INT8/FP16, and FP32 produced highly similar text while faster-whisper preserved more
punctuation and slightly more text than openai-whisper.

## RTX A4000 Laptop comparison

System: ZBookFuryG8, RTX A4000 Laptop GPU 8GB (Ampere, compute capability 8.6), driver 595.95,
torch 2.6.0+cu124, Python 3.13.13, openai-whisper 20250625, faster-whisper 1.2.1.
Same first 300 seconds of the same Spanish recording as above.

| Backend | Config | T1000 | RTX 4070 | RTX A4000 |
|---|---|---:|---:|---:|
| openai-whisper medium | fp16 | 109.2s | 20.9s | 33.2s |
| faster-whisper medium | float16 | 119.9s | 10.9s | **15.6s** |
| faster-whisper medium | int8_float16 | 40.9s | 10.6s | 18.2s |
| faster-whisper medium | float32 | — | 13.7s | 22.5s |

The A4000 is ~7.7x faster than the T1000 on `float16` and ~3.3x on openai-whisper `medium`.
It is slower than the 12 GB RTX 4070 as expected, but confirms the tensor-core pattern:
`float16` beats `int8_float16` on Ampere/Ada, while the non-tensor-core T1000 prefers `int8_float16`.
Results are in `Docs/transcription-benchmark-results.csv` (machine `ZBookFuryG8`).

## Summarization benchmark — 2026-08 (Unitec webinars)

Two Spanish webinars (~91-99k chars / ~16.6k words) were transcribed with Whisper `medium` and
summarized with cloud and local models. See transcripts in `Output/transcripts/` (git-ignored).

| Provider | Model | Method | Quality |
|---|---|---|---|
| OpenAI | `gpt-5.6-luna` | single-shot | Most detailed / source-faithful (5,530 words, 1,119 lines on 2026-08-25); preserves caveats and long-tail topics |
| Google | `gemini-flash-latest` | 7-8 × 15k map-reduce | Best concise executive summary (2,753 words); strong price/bonus density but some reduction conflations (e.g. merged Peru testimonials, pricing example) |
| Local | `gemma4:e4b-it-qat` 6.1 GB (QAT) | 32 × 3k map-reduce | Spanish-capable, 100% GPU 3.1 GB on 8 GB A4000; smoke test 6k chars → 73.6s (24.5s/chunk, ~13 min projected full) vs `gemma4:12b` 7.6 GB at 31%/69% CPU/GPU (~51 min/chunk, ~27 h full). Output shallower than cloud, occasional entity errors |
| Local | `gemma4:12b` 7.6 GB | 32 × 3k map-reduce | Good quality when run but CPU offload makes it impractical on 8 GB |

The `summarize.py` chunker was fixed to hard-wrap long Whisper stretches without sentence boundaries
(`scripts/Python/summarize.py:51-74`), otherwise a 99k-char transcript produced only 2 chunks and
silently dropped context. Local Ollama also defaults to 4096 context; the 32-chunk reduce prompt
exceeds that and needs `num_ctx` ≥16384 or batched hierarchical reduce.

Decision: keep **OpenAI (`gpt-5.6-luna`) and Gemini (`gemini-flash-latest`)** for production summaries.
Local Gemma is retained for experimentation/offline fallback only.

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
timing table). It also updates `Docs/transcription-benchmark-results.csv`, which is the shared,
portable dataset for comparing machines. Commit or copy that CSV between machines before running
the benchmark so each laptop adds its results. Use `--machine-name` to override the OS hostname.

When comparing GPUs, note especially:

- GPU model, VRAM, compute capability, driver version
- which `compute_type` won: try `float16` vs `int8_float16` — tensor-core GPUs usually prefer
  `float16`, non-tensor-core Turing (T1000/GTX 16-series) prefers `int8_float16`
- whether anything fell back to CPU (chars/time vs the T1000 reference above is a good sanity check)
- compare transcription time separately from model load time; first use may include model downloads,
  while later runs use the local cache
- use the same audio, 300-second slice, language, model, and beam size for meaningful comparisons

## Follow-up

Switch `scripts/Python/transcribe.py` from openai-whisper to faster-whisper, selecting FP16 on
modern RTX GPUs and INT8/FP16 on the T1000. Summarization stays on Gemini/OpenAI (best detail
retention); local Gemma is for experimentation only.
