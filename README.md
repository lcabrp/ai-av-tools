# Growth Partners

Extract insights from audio/video files using local or cloud AI models.

## Setup

### Prerequisites

- Python 3.13+ (managed via `uv`)
- ffmpeg (for audio/video processing, installable via `winget install --id Gyan.FFmpeg`)
- Ollama (for local models) OR API keys (OpenAI, Anthropic, Google Gemini)

### Installation

```bash
uv sync
```

### Local Models (GPU recommended)

1. Install [Ollama](https://ollama.com)
2. Pull models:
```bash
ollama pull gemma4:12b
ollama pull gemma4:e4b
```

### Cloud Models & API Keys

1. Copy `.env.example` to `.env`
2. Add your API keys as needed:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

### Complete End-to-End Pipeline (Transcribe + Summarize)

Process a YouTube video, a single video/audio file, or an entire directory of media files:

```bash
# Process a YouTube video directly (uses captions when available, then summarizes with OpenAI Luna)
uv run main.py "https://youtu.be/HzGOWq5UyjY"

# Process a local video or audio file (or folder); Luna is the default summary model
uv run main.py Docs/Video/JaimeHiguera

# Opt into provider fallback (OpenAI -> Anthropic -> Gemini -> Ollama)
uv run main.py Docs/Video/JaimeHiguera --provider auto
```

### YouTube Video Transcripts

Instantly extract YouTube subtitles/transcripts without needing local GPU audio transcription:

```bash
# Standalone YouTube transcript extraction
uv run scripts/Python/youtube_transcript.py "https://youtu.be/HzGOWq5UyjY"

# Optional: specify custom output directory or language
uv run scripts/Python/youtube_transcript.py "https://youtu.be/HzGOWq5UyjY" --output-dir Output/transcripts --language es
```

This generates:
- `<Title> [<ID>].txt`: Continuous plain text formatted for LLM summarization.
- `<Title> [<ID>]_timestamps.txt`: Human-readable transcript with `[MM:SS]` timestamps.
- `<Title> [<ID>].lang`: Detected language metadata.

Caption language is auto-selected: English tracks are preferred, then any other available
track (e.g. the original-language auto-generated captions). This matters because
`YouTubeTranscriptApi.fetch()` only requests `en` by default, so a Spanish-only video used to
look "subtitle-less" and silently drop into the slow Whisper path. The detected language is
written to the `.lang` file, which makes the summary come out in the video's own language by
default — pass `--language english` when you'd rather have it in English.

If a video has subtitles disabled, `main.py` automatically falls back to downloading the
audio with `yt-dlp` (saved under `Docs/Audio/youtube/`, skipped if already downloaded) and
transcribing it locally with faster-whisper before summarizing — so the end-to-end pipeline
works for any YouTube URL.

### Transcribe Audio/Video

Supports `.wav`, `.mp3`, `.m4a`, `.mp4`, `.mov`, `.mkv`, etc., for single files or entire folders:

```bash
uv run scripts/Python/transcribe.py Docs/Video/JaimeHiguera
```

Detects language automatically and saves transcripts to `Output/transcripts/`. Default model is
`medium` — noticeably more accurate than `base` for real conversations (mumbled speech, accents,
crosstalk), which matters because a garbled transcript can't be recovered by even the best
summarization model. Use `--model large` for maximum accuracy if you can tolerate the extra runtime.

To benchmark faster-whisper compute types (FP16, INT8/FP16, FP32) on a given GPU, see
[`Docs/whisper-benchmark.md`](Docs/whisper-benchmark.md) and `scripts/Python/benchmark_transcription.py`.
Current benchmark results recommend faster-whisper `medium`: FP16 on modern RTX GPUs, or
INT8/FP16 on the NVIDIA T1000. Cross-machine results are tracked in
[`Docs/transcription-benchmark-results.csv`](Docs/transcription-benchmark-results.csv).

### Summarize Transcripts

Summarize single transcript files or all transcripts in a directory using your choice of LLM provider:

```bash
# Default: OpenAI Luna, single-shot
uv run scripts/Python/summarize.py Output/transcripts/sample.txt

# Specific providers
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --provider ollama
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --provider openai --openai-model gpt-6-luna
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --provider anthropic --anthropic-model claude-sonnet-4-6
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --provider gemini --gemini-model gemini-flash-latest
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --provider auto

# Override summary language
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --language english
```

Summaries are saved as `Output/<name>_summary_<provider>.md` (e.g. `_summary_gemini.md`,
`_summary_openai_luna.md`) so results from different providers stay side-by-side instead of
overwriting each other. The default provider is OpenAI Luna, which summarizes in one request.

#### Provider policy: Luna first, one summary per recording

**Luna (`gpt-6-luna`) is the default and normally the only provider we run.** One summary per
recording — do not also generate a Gemini/Anthropic/Ollama summary "just in case", since that
doubles cost and leaves two competing documents downstream.

Gemini (`gemini-flash-latest`) is a **fallback only**, used when Luna is genuinely unavailable
(missing/invalid `OPENAI_API_KEY`, or the OpenAI request fails after retrying). When that happens,
say so explicitly in the run report. Other providers are reserved for explicit side-by-side
comparisons and benchmarks (see `scripts/Python/compare_models.py`). See `AGENTS.md` for the full
operating rules.
Use `--provider auto` to opt into provider fallback (OpenAI -> Anthropic -> Gemini -> Ollama);
in auto mode the tag reflects whichever provider actually succeeded. Passing an explicit
`--output` file path overrides the tag.

#### Gemini summaries and automatic chunking

Gemini is available as an explicit provider or through `--provider auto`. In side-by-side comparisons against
Ollama (`gemma4:12b`/`e4b`), OpenAI (`gpt-5.6-sol/terra/luna`), and Anthropic (`claude-sonnet-4-6`),
it consistently retained the most concrete detail (prices, names, step-by-step instructions).

Long transcripts (> `--gemini-chunk-chars`, default **15000** characters) are **automatically**
summarized using map-reduce chunking: each chunk is summarized individually, then merged into one
final document. This measurably improves detail retention on long/dense sessions — in testing, it
surfaced multiple individually-named cases and verbatim step-by-step instructions that a single-shot
call missed entirely, at the cost of a handful of extra API calls (e.g. ~9 calls for a 2-hour session).
Short transcripts skip chunking automatically since they fit in one call.

```bash
# Tune the chunk-size threshold (smaller = more calls, marginally safer for very dense content)
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --provider gemini --gemini-chunk-chars 15000
```

#### Local models (Ollama) and low-VRAM machines

For the local Ollama fallback (`gemma4:12b` primary, `gemma4:e4b` secondary), use `--chunked` to
apply the same map-reduce approach. This matters much more for local models than for Gemini, since
small/quantized models have limited effective context and silently drop sections on long transcripts
otherwise.

```bash
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --provider ollama --chunked --chunk-chars 3000
```

**Finding the right `--chunk-chars` for your GPU** (do this once per machine):
1. Check the model's disk size with `ollama list` — this is roughly how much VRAM the weights need.
2. Run a chunk and check `ollama ps` — it reports a GPU/CPU % split. Any CPU % means the current
   chunk size (prompt + context) doesn't fully fit in VRAM and is slowing down.
3. Adjust `--chunk-chars` up/down until `ollama ps` shows ~100% GPU, then back off ~10-15% for safety.
4. Rough starting points by VRAM: **8GB** -> `gemma4:12b` at ~3000-4000 chars; **6GB** -> prefer
   `gemma4:e4b` at ~2000-2500 chars; **4GB** -> `gemma4:e4b` only, ~1000-1500 chars (or just rely on
   Gemini instead of local).

Bigger chunks aren't automatically better for local models — in testing, `gemma4:12b` on an 8GB GPU
actually produced a *less* complete summary at 5000 chars/chunk than at 3000, since there was very
little VRAM headroom left over after the model weights.

### Multi-Model Benchmark Comparison

Compare generated summaries from **Ollama** (`gemma4:12b`), **OpenAI** (`gpt-5.6-sol/terra/luna`), **Anthropic** (`claude-sonnet-4-6`), and **Google** (`gemini-flash-latest`) side-by-side without overwriting:

```bash
uv run scripts/Python/compare_models.py Output/transcripts/sample.txt
```

## Project Structure

```
Docs/Audio/          # Audio files (not tracked)
Docs/Video/          # Video files (not tracked)
Output/
  transcripts/       # Generated transcripts
  *_summary_*.md     # Markdown summaries, tagged per provider
scripts/Python/      # Transcribe, youtube_transcript, summarize, and compare_models scripts
main.py              # Main end-to-end pipeline CLI
```

## Models Supported

- **Transcription**: 
  - **YouTube**: `youtube-transcript-api` (instant subtitle extraction, zero-GPU overhead)
  - **Local Media**: faster-whisper (local CUDA-accelerated, supports all audio and video formats; default model `medium`, auto FP16 on modern RTX / INT8-FP16 on older GPUs)
- **Summarization**:
  - **Default**: OpenAI `gpt-6-luna` (single-shot)
  - **Local**: Ollama `gemma4:12b`, `gemma4:e4b`
  - **OpenAI**: `gpt-6-luna`, `gpt-6-sol`, `gpt-6-astra`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`
  - **Anthropic**: `claude-sonnet-4-6`
  - **Google**: `gemini-flash-latest`

## Notes on Summary Quality

- **Prompting**: all providers share a detail-preserving system prompt (see `build_system_prompt` in
  `scripts/Python/summarize.py`) that explicitly instructs the model to keep numbers, names, tool/platform
  names, and step-by-step instructions rather than compressing them into vague generalizations. This helps
  smaller/local models the most, since they compress more aggressively than frontier cloud models by default.
- **`.env` loading**: `summarize.py` loads `.env` from an explicit path (project root), rather than relying on
  `python-dotenv`'s stack-frame-based auto-discovery, which was found to unpredictably skip loading depending
  on which script imported the module first (e.g. running `compare_models.py` directly vs. importing it).
- **API keys**: if a provider fails with an auth error, double check the key is still valid (e.g. Anthropic
  keys can be revoked/regenerated in the [Anthropic Console](https://console.anthropic.com/settings/keys)).
  Workload identity federation (short-lived cloud-issued tokens) is not applicable here since this project
  runs as local scripts, not from a CI/cloud provider with an identity provider.
