# av-tools

Extract insights from audio/video files using local or cloud AI models.

## Setup

### Prerequisites

- Python 3.13+ (managed via `uv`)
- ffmpeg (for audio/video processing, installable via `winget install --id Gyan.FFmpeg`)
- Deno (JavaScript runtime that `yt-dlp` needs for YouTube extraction — see
  [Download Audio from YouTube](#download-audio-from-youtube-mp3--m4a--opus))
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

### Download Audio from YouTube (MP3 / M4A / Opus)

A 4KVideoDownloader-style downloader: give it one or more URLs (or a playlist) and it saves
tagged audio with embedded cover art.

```bash
# Default: M4A (bit-exact copy of YouTube's AAC), saved to Output/downloads/m4a/
uv run scripts/Python/download_audio.py "https://youtu.be/VIDEO_ID"

# MP3, if a device/car stereo needs it
uv run scripts/Python/download_audio.py "https://youtu.be/VIDEO_ID" --preset mp3

# Several videos or a whole playlist at once
uv run scripts/Python/download_audio.py URL1 URL2 URL3
uv run scripts/Python/download_audio.py "https://www.youtube.com/playlist?list=..." --playlist

# Batch from a text file (one URL per line, blank lines and # comments ignored)
uv run scripts/Python/download_audio.py --batch-file urls.txt --preset mp3-vbr

# Preview titles/durations without downloading anything
uv run scripts/Python/download_audio.py URL --dry-run

# Cut silent padding off the end of each file (lossless)
uv run scripts/Python/download_audio.py URL --trim-silence

# Prefer a specific track on a multi-language / auto-dubbed video (optional)
uv run scripts/Python/download_audio.py URL --audio-lang es

# Age-restricted or "confirm you're not a bot" videos
uv run scripts/Python/download_audio.py URL --cookies-from-browser chrome
uv run scripts/Python/download_audio.py URL --cookies path\to\cookies.txt   # if the browser file is locked
```

If YouTube answers "Sign in to confirm you're not a bot" it is usually a temporary block on your
IP after many requests (e.g. a big playlist): waiting a while is enough. To get past it sooner,
pass your logged-in browser's cookies. Chrome and Edge lock their cookie file while running (they
also keep background processes after you close the window), so quit them completely first, or use
`--cookies` with a `cookies.txt` exported from the browser.

Files land in `Output/downloads/<preset>/` (git-ignored; change it with `--output-dir`) as `<title> [<video_id>].<ext>` with title,
artist, and date tags plus the thumbnail as cover art. A `.downloaded.txt` archive per preset
skips videos you already fetched (`--no-archive` forces a re-download). Because it uses
`yt-dlp`, it also works on other supported sites (SoundCloud, Vimeo, ...), not just YouTube.
YouTube auto-dubbing can expose many audio tracks, and occasionally the default one is broken
(a few seconds of sound, then silence, repeating). You don't need any flag for that: every
download is scanned for silence, and a mostly-silent track is automatically swapped for the first
clean alternative track. Silence at the very start or end of a video is only reported as a note,
because the upload itself is padded and no track can fix it. Add `--trim-silence` to cut that
trailing silence (5 s or longer) off each file: it is a stream copy, so the audio, tags and cover
art are untouched, and 0.5 s is kept after the last sound. `--audio-lang es` (or `en`, `es-MX`)
only sets a preference for genuinely multi-language videos; it is optional.

Presets:

| Preset | Output | Notes |
|---|---|---|
| `m4a` (default) | AAC | **Bit-exact copy of YouTube's audio** — no re-encode, no quality loss, plays on Windows/macOS/iOS/Android and modern car stereos |
| `mp3` | MP3 320 kbps CBR | Re-encodes the lossy source: bigger *and* slightly worse. Only for MP3-only gear (old car stereos, some DJ software) |
| `mp3-vbr` | MP3 LAME V0 (~245 kbps) | Best MP3 quality-per-byte when you do need MP3 |
| `opus` | Opus | True stream copy of YouTube's Opus stream — smallest file, highest fidelity per byte |
| `flac` | FLAC | Lossless container of the decoded source (large, no gain over the source) |
| `wav` | WAV 16-bit PCM | Uncompressed, for editing |
| `native` | Original stream (`.webm`/`.m4a`) | No conversion at all — the true "no quality loss" option |

**About bitrate ("I want 320 kbps"):** YouTube has no 320 kbps stream. Its best audio is AAC
~128-130 kbps or Opus ~106-160 kbps, so a 320 kbps MP3 is a padded re-encode of a lossy source —
3× the size for audio slightly *worse* than the original. The default `m4a` preset avoids the
whole problem by copying the source stream bit-for-bit, which is the highest quality that exists
for that video. Choose `mp3` only for device compatibility, not for quality.

#### Requirements on a new PC (ffmpeg + Deno)

Two tools must be on `PATH` before the downloader works properly. Neither ships with the repo:

```powershell
winget install --id Gyan.FFmpeg      # required: conversion, remuxing, tags
winget install --id DenoLand.Deno    # required: YouTube extraction by yt-dlp
```

Then **open a new terminal** — `winget` edits `PATH`, and an already-running shell will not see
it. Verify both:

```powershell
ffmpeg -version    # should print a version banner
deno --version     # should print "deno 2.x.x"
```

Why Deno: YouTube's web player ships obfuscated JavaScript that computes signature and `n`
parameters, which are embedded in the real media URLs. yt-dlp no longer bundles its own JS
interpreter, so it uses a JS runtime (Deno by default) to run that code. It is used only for a
moment during extraction — not a background service, and it is not involved in converting or
transcribing anything. Without it, yt-dlp prints:

```
WARNING: [youtube] No supported JavaScript runtime could be found...
some formats may be missing
```

Downloads may still succeed, but often with fewer/lower-quality formats available, and they
break sooner when YouTube changes its player.

Deno is what yt-dlp enables on its own. If you only have Node or Bun, yt-dlp does *not* pick it
up automatically — so `download_audio.py` detects `deno`/`node`/`bun` on `PATH` and passes
whichever it finds to yt-dlp, and prints the hint above when none exist. Any of the three works;
Deno is the recommended one because it is yt-dlp's default and best tested.

Install it only if you plan to download from YouTube — Whisper transcription and summarization
do not need it.

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

#### Language: summarize in the recording's language, never translate

Recordings are summarized in **Spanish or English**, matching the source — we never translate.
`summarize.py` picks the language from the `.lang` file written at transcription time (or
auto-detects it from the transcript when that file is missing), then verifies the generated
summary with a dependency-free detector:

- If the summary is in the expected language, the run prints `Language check: summary is in spanish.`
- If it drifts (e.g. a Spanish transcript summarized in Chinese), it prints a warning, **regenerates
  once** to recover the original language, and re-checks. A persisting mismatch is saved but logged
  with `status=lang-mismatch` in `Docs/operations-log.csv` so it can be spotted later.

No translation is ever requested, so a Spanish recording always produces a Spanish summary.
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
Docs/Audio/          # Audio files (not tracked); downloads/ holds MP3/M4A/Opus output
Docs/Video/          # Video files (not tracked)
Output/
  transcripts/       # Generated transcripts (staging: move them to exploring-pathways)
  *_summary_*.md     # Markdown summaries, tagged per provider (staging)
scripts/Python/      # Transcribe, youtube_transcript, download_audio, summarize, file_outputs, compare_models
main.py              # Main end-to-end pipeline CLI
```

`Output/` is staging only: finished runs are filed by topic into the downstream repos —
`ky-tax-preparer/course-archives/<Brand>/<Clase|Date>/` for tax-preparer recordings, and
`exploring-pathways/Pathways/<Topic>/<Brand>/<Clase|Date>/` for everything else — using
`scripts/Python/file_outputs.py`, so `Output/` normally holds nothing but `.gitkeep`.

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
