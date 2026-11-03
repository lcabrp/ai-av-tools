# Agent Notes

Operational rules for agents working in this repo. Keep changes consistent with these.

## Environment

- Windows + PowerShell. Use `uv` for everything; never `pip`.
  - `uv sync --locked`, `uv run python -u ...`, `uv run main.py ...`
- The project root is the working directory for all commands.
- API keys live in `.env` at the repo root (git-ignored): `OPENAI_API_KEY`, `GEMINI_API_KEY`,
  and optionally `ANTHROPIC_API_KEY`.

## Summarization provider policy

**OpenAI Luna (`gpt-6-luna`) is the default and the only summary provider we run.**

- Run **one** summary per recording, with Luna. Do not also run Gemini/Anthropic/Ollama on the
  same transcript "just in case" — that doubles cost and produces two competing documents.
- Use Gemini (`gemini-flash-latest`) **only as a fallback**, when Luna is genuinely unavailable:
  - `OPENAI_API_KEY` missing/invalid, or
  - the OpenAI request fails after retrying (model unavailable, hard upstream error).
  When falling back, say so explicitly in the report (which provider was used and why).
- Use another provider only when the user explicitly asks for a comparison or a benchmark run
  (e.g. `scripts/Python/compare_models.py`, `Docs/local-summary-benchmark-results.csv`).
- Both CLIs already default to `gpt-6-luna`; pass `--openai-model` only to override deliberately.

## Standard workflow for a local audio/video file

1. Verify the file exists and get its duration with `ffprobe`.
2. Transcribe:
   `uv run python -u scripts/Python/transcribe.py "<path>" --model medium`
   → `Output/transcripts/<name>.txt` (faster-whisper `medium`, CUDA FP16 on this machine).
3. Summarize with Luna:
   `uv run scripts/Python/summarize.py "Output/transcripts/<name>.txt" --provider openai --openai-model gpt-6-luna`
   → `Output/<name>_summary_luna.md` (or `_summary_openai_luna.md` if `--output` is omitted).
4. Validate: check the file exists and is non-trivial (chars/words/lines), then read the first
   ~20 lines to confirm the content matches the recording.
5. Report: source, duration, detected language, transcript size, summary path + size, and a
   short topic summary. Mention any anomaly (mis-detected language, truncated audio, retries).

## YouTube URLs

`uv run main.py "<url>" --provider openai --language es`

- `main.py` prefers YouTube captions; when none exist it downloads audio with yt-dlp to
  `Docs/Audio/youtube/` and transcribes locally, then summarizes.
- `main.py` writes `_summary_<provider>.md` tags; for Luna that is `_summary_openai_luna.md`.
- Whisper sometimes mis-detects Spanish as `zh` on certain channels — the transcript content is
  still Spanish; note it rather than re-running blindly.
- If Luna's output language drifts (e.g. a Spanish transcript summarized in Chinese), report it and
  offer a re-run with an explicit language.

## Outputs and tracking

- `Output/transcripts/` is git-ignored; `Output/*_summary_*.md` are local artifacts.
- `Docs/operations-log.csv` is tracked and appended to by the summarize/transcribe tooling;
  commit it together with doc changes when relevant.
- Commit document/tooling changes with clear messages; don't commit generated media or
  transcripts.
