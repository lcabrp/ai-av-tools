# Whisper Transcription Benchmark: openai-whisper vs faster-whisper

Why this exists: `scripts/Python/transcribe.py` uses `faster-whisper` `medium` (switched from
openai-whisper after these benchmarks) because `base` was not accurate enough for real
conversations. `faster-whisper` claims ~4x speedups but initially appeared no faster here —
that turned out to be a flawed test (see "The CPU trap" below).

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

`scripts/Python/transcribe.py` has since moved to `faster-whisper` too (see *Follow-up (done)*
below), and `benchmark_transcription.py` now drives only `faster-whisper`.

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

## RTX A2000 Laptop comparison

System: ZBookPowerG9-03, RTX A2000 8GB Laptop GPU (Ampere, compute capability 8.6, 20 SMs),
driver 595.95, torch 2.6.0+cu124, Python 3.13.12, openai-whisper 20250625, faster-whisper 1.2.1.
Same first 300 seconds of the same Spanish recording (local filename
`KTA_Institute_TaxPreparer_2026-08-26.mp3`; identical character counts to the other machines —
e.g. 3794 chars on faster-whisper `float16`, 3555 on openai-whisper — confirm same audio content).

| Backend | Config | T1000 | RTX 4070 | RTX A4000 | RTX A2000 |
|---|---|---:|---:|---:|---:|
| openai-whisper medium | fp16 | 109.2s | 20.9s | 33.2s | 26.9s |
| faster-whisper medium | float16 | 119.9s | 10.9s | **15.6s** | 20.1s |
| faster-whisper medium | int8_float16 | 40.9s | 10.6s | 18.2s | **19.7s** |
| faster-whisper medium | float32 | — | 13.7s | 22.5s | 35.8s |

The A2000 is ~6.0x faster than the T1000 on `float16` and ~4.1x on openai-whisper `medium`.
It sits where expected in the lineup: ~1.9x slower than the RTX 4070 and ~1.26x slower than the
A4000 (best-config comparison), ~2.1x faster than the T1000. On this machine faster-whisper FP16
(20.1s) is ~1.34x faster than openai-whisper FP16 (26.9s).
Note: `int8_float16` (19.7s) edged out `float16` (20.1s) by ~0.4s — effectively a tie, but a mild
break from the tensor-core pattern seen on the RTX 4070/A4000 where `float16` won. FP16 remains the
conservative default for compute capability >= 8.0 since the gap is within run-to-run noise and it
avoids quantization.
Results are in `Docs/transcription-benchmark-results.csv` (machine `ZBookPowerG9-03`).

## RTX 4060 Laptop comparison

System: Legion7-16IRX9, NVIDIA GeForce RTX 4060 Laptop GPU 8GB (Ada, compute capability 8.9,
24 SMs), driver 616.56, torch 2.6.0+cu124, Python 3.13.7, openai-whisper 20250625,
faster-whisper 1.2.1. Same first 300 seconds of the same Spanish KTA Institute recording.

| Backend | Config | RTX 4070 | RTX 4060 | RTX A4000 | RTX A2000 |
|---|---|---:|---:|---:|---:|
| openai-whisper medium | fp16 | 20.9s | 21.9s | 33.2s | 26.9s |
| faster-whisper medium | float16 | **10.9s** | 14.2s | 15.6s | 20.1s |
| faster-whisper medium | int8_float16 | **10.6s** | 14.4s | 18.2s | 19.7s |
| faster-whisper medium | float32 | 13.7s | 23.2s | 22.5s | 35.8s |

The RTX 4060 processed the 300-second sample at about **21.1x real-time** with faster-whisper
FP16. It was about **1.10x faster than the A4000** and **1.42x faster than the A2000** in this
test, while the RTX 4070 remained faster. FP16 and INT8/FP16 were effectively tied on the 4060
(14.2s vs. 14.4s); FP16 remains the conservative default. The FP16 transcript had 3,794
characters, matching the RTX 4070 result and closely matching the A2000/A4000 runs. Character
counts are only a rough output comparison, not an accuracy score.

Results are in `Docs/transcription-benchmark-results.csv` (machine `Legion7-16IRX9`); the
benchmark report and per-configuration transcripts are saved under
`bench_output/RTX4060-KTA-2026-10-25/` (git-ignored).

## Intel Arc Pro B70 via OpenVINO — 2026-09-20

System: HP-Z2-G9, Intel Arc Pro B70 dGPU (driver 32.0.101.8805) + i9-12900, **no NVIDIA
GPU**. torch reports `cuda_available=False`, so the CUDA backends don't apply. The Intel path is
OpenVINO (2026.4.0) + optimum-intel: export once with `optimum-cli`, then run
`OVModelForSpeechSeq2Seq` on `GPU.1` (the Arc B70; `GPU.0` is the UHD 770 iGPU — verify with
`Core().get_property(d, 'FULL_DEVICE_NAME')`). IPEX is not an option on Windows (no `win_amd64`
wheels for >= 2.6).

Audio: `MargaritaPasos_01.mp3` (96.5 min Spanish livestream) — a different file from the 300s
reference slices, so cross-GPU comparison below uses throughput (audio-sec per wall-sec).

| Slice | Transcribe | Throughput | Chars |
|---|---|---:|---:|
| 60s | 6.4s | 9.4x | 828 |
| 300s | 32.7s | 9.2x | 4014 |
| full 5789s | 805.8s (13.4 min) | 7.2x | 78,526 (`bench_output/openvino-whisper-medium-gpu1-full.txt`) |

CPU baseline on the same machine (i9-12900, 60s slice): openai-whisper medium fp32 49.9s
(852 chars), faster-whisper medium int8 26.2s (796 chars). The Arc B70 via OpenVINO is ~4x
faster than the best CPU option with on-par quality (same opening sentence, similar char counts).

Throughput lineup (300s-slice numbers, best faster-whisper config except Arc = OpenVINO):

| GPU | Time | Throughput |
|---|---|---:|
| RTX 4070 | 10.6s | 28x |
| RTX A4000 Laptop | 15.6s | 19x |
| RTX A2000 Laptop | 19.7s | 15x |
| RTX A1000 Laptop | 20.3s | 15x |
| **Arc Pro B70** | **32.7s** | **9.2x** |
| T1000 | 40.9s | 7.3x |

The Arc B70 lands between the T1000 and the Ampere laptop parts — a viable transcription GPU,
roughly 1.25x faster than the T1000's best config.

Verdict: no meaningful gain — slower than every NVIDIA option already benchmarked, and it
needs a separate OpenVINO pipeline outside `benchmark_transcription.py`/`transcribe.py`.
Not worth adopting for transcription; staying with the NVIDIA lineup.

Reproduce:

```bash
# one-time export (~3 GB of IR; keep the output dir, re-export is slow)
uv run --with openvino --with "optimum[openvino]" optimum-cli export openvino \
  --model openai/whisper-medium --task automatic-speech-recognition <ov-model-dir>

# transcribe on the Arc dGPU (naive 30s non-overlapping chunks; timings are the point —
# for production use chunking with overlap or the transformers chunked pipeline instead)
uv run --with openvino --with "optimum[openvino]" --with librosa --with soundfile \
  python bench_ov.py <audio> GPU.1
```

Caveats: the OV runs don't go through `benchmark_transcription.py` (CUDA-oriented script);
their rows were appended to `Docs/transcription-benchmark-results.csv` manually with
`backend=openvino-whisper`. The `attention_mask` warning during `generate()` is harmless for
benchmark purposes (char counts match the other backends).

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

## Local SLM experiment — 2026-09 (AGP FAQ call)

Question: can a small local model do this job and save money? Test: the 98k-char Spanish FAQ
transcript on the RTX 4070 with an improved pipeline (native Ollama endpoint, temperature 0.1,
32K context for reductions, hierarchical map-reduce, attribution-preserving prompts).
Results in `Docs/local-summary-benchmark-results.csv`, outputs in `Output/local-summary-benchmarks/`.

| Model | Time | Words | Named cases (/17) | Score |
|---|---|---:|---:|---:|
| `ministral-3:3b` | 6.4 min | 1,448 | 2 | 2/10 |
| `phi4-mini` | 4.4 min | 355 | 0 | 2/10 |
| `alibayram/smollm3` (community package) | 5.0 min | 684 | 0 | 1/10 |
| `gemma4:12b` (old pipeline) | 38.5 min | 0 | 0 | fail |
| `qwen3.5:9b` (old pipeline) | 26.9 min | 375 | — | fail |

Findings:

- All local runs lost nearly all participant attribution, numbers, and dated action items.
  Ministral kept 2 of 17 named cases; Phi-mini and SmolLM3 kept none.
- SmolLM3 invented a nonexistent 4-pillar "PDD" program; Ministral fossilized ASR errors
  ("chisuit", "micho") into entities and duplicated sections (map-reduce stitching artifacts).
- Longer output did not mean better recall: Ministral's 1,448 words still missed all metrics
  and every dated commitment.
- Pipeline lesson: alternating `num_ctx` per call forces Ollama model reloads — use one uniform
  context; cap extraction output so reductions stay shallow (40k-char groups → 1 level, 3 groups).

Verdict: local SLMs are **not viable replacements** for cloud summaries on dense meeting content.
They stay as offline fallback only. Production remains Gemini/Luna.

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
(`winget install --id Gyan.FFmpeg`). For Intel Arc GPUs (no CUDA), use the OpenVINO path
described in the Arc Pro B70 section instead.

```bash
git clone <this repo> && cd av-tools
uv sync                # installs torch from the cu124 index + faster-whisper

# 1) Verify GPU is visible to torch (must say True and show the GPU name)
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2) Quick benchmark: first 300s of the file (same settings as the T1000 reference run)
uv run scripts/Python/benchmark_transcription.py "E:\path\to\TaxPreparer_KTA_Institute_2026-08-26.mp3" --slice 300 --language es

# 3) Full-file benchmark (takes a while)
uv run scripts/Python/benchmark_transcription.py "E:\path\to\file.mp3" --slice 0 --language es

# Optional: benchmark a different model size (tiny/base/small/medium/large)
uv run scripts/Python/benchmark_transcription.py file.mp3 --model large
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

## Follow-up (done)

`scripts/Python/transcribe.py` now uses faster-whisper, auto-selecting FP16 on GPUs with
compute capability >= 8.0 and INT8/FP16 on older GPUs such as the T1000
(`--compute-type` / `--device` override when needed). Summarization stays on Gemini/OpenAI
(best detail retention); local SLMs are for experimentation only.
