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
ollama pull gemma4:e4b
ollama pull gemma4:e2b
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

Process a single video/audio file or an entire directory of media files:

```bash
# Process a video or audio file (or folder) with smart automatic model fallback
uv run main.py Docs/Video/JaimeHiguera

# Specify whisper model and summary provider explicitly
uv run main.py Docs/Video/JaimeHiguera --whisper-model base --provider auto
```

### Transcribe Audio/Video

Supports `.wav`, `.mp3`, `.m4a`, `.mp4`, `.mov`, `.mkv`, etc., for single files or entire folders:

```bash
uv run scripts/Python/transcribe.py Docs/Video/JaimeHiguera
```

Detects language automatically and saves transcripts to `Output/transcripts/`.

### Summarize Transcripts

Summarize single transcript files or all transcripts in a directory using your choice of LLM provider:

```bash
# Auto mode: Ollama (e4b -> e2b) -> Anthropic -> Gemini -> OpenAI
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --provider auto

# Specific providers
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --provider ollama
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --provider openai --openai-model gpt-4o
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --provider anthropic --anthropic-model claude-sonnet-4-6
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --provider gemini --gemini-model gemini-flash-latest

# Override summary language
uv run scripts/Python/summarize.py Output/transcripts/sample.txt --language english
```

### Multi-Model Benchmark Comparison

Compare generated summaries from **Ollama (Gemma)**, **OpenAI (GPT-4o / o3-mini / GPT-4o-mini)**, **Anthropic (Claude Sonnet)**, and **Google (Gemini Flash)** side-by-side without overwriting:

```bash
uv run scripts/Python/compare_models.py Output/transcripts/sample.txt
```

## Project Structure

```
Docs/Audio/          # Audio files (not tracked)
Docs/Video/          # Video files (not tracked)
Output/
  transcripts/       # Generated transcripts
  *.md               # Markdown summaries
scripts/Python/      # Transcribe, summarize, and compare_models scripts
main.py              # Main end-to-end pipeline CLI
```

## Models Supported

- **Transcription**: OpenAI Whisper (local CUDA-accelerated, supports all audio and video formats)
- **Summarization**:
  - **Local**: Ollama `gemma4:e4b`, `gemma4:e2b`
  - **OpenAI**: `gpt-4o`, `o3-mini`, `gpt-4o-mini`
  - **Anthropic**: `claude-sonnet-4-6`
  - **Google**: `gemini-flash-latest`
