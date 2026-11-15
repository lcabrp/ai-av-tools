# Agent Notes

Operational rules for agents working in this repo. Keep changes consistent with these.

## Environment

- Windows + PowerShell. Use `uv` for everything; never `pip`.
  - `uv sync --locked`, `uv run python -u ...`, `uv run main.py ...`

## Available tools on this PC

The full list (with usage notes) is in the global OpenCode instructions at
`~/.config/opencode/instructions.md`. Summary of the non-obvious ones:

- **Search/navigation:** `rg` (text), `ast-grep` (syntax-aware), `fd` (files), `bat` (preview), `fzf` (fuzzy).
- **Data:** `jq` (JSON), `duckdb` (CSV/Parquet/Arrow — never read raw into context), `sqlcmd` (SQL Server).
- **Python:** `uv` (never pip), `ruff` (lint/format), `nbstripout` (notebooks).
- **GitHub:** `gh` (CLI).
- **Node/JS:** `node`, `npm`, `repomix` (context packaging).
- **Media:** `ffmpeg` (audio/video), `deno` (JS runtime for yt-dlp).
- **Shell helpers:** `rtk` (command prefix), `rtk proxy <cmd>` for unsupported commands.
- The project root is the working directory for all commands.
- API keys live in `.env` at the repo root (git-ignored): `OPENAI_API_KEY`, `GEMINI_API_KEY`,
  and optionally `ANTHROPIC_API_KEY`.
- External binaries must be on `PATH` (installed once per machine, never via `uv`):
  - `ffmpeg` — audio/video conversion, remuxing, tagging: `winget install --id Gyan.FFmpeg`
  - `deno` — JS runtime `yt-dlp` uses to extract YouTube URLs:
    `winget install --id DenoLand.Deno`
  `winget` edits `PATH`, so **restart the shell** afterwards. A stale shell does not fail loudly —
  yt-dlp just warns `No supported JavaScript runtime could be found` and silently offers fewer
  formats. Verify with `ffmpeg -version` and `deno --version` on a new machine.

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

## Language policy: summarize in the source language

Recordings are Spanish or English. **Summaries are never translated** — a Spanish recording gets a
Spanish summary, an English one gets English.

- Language comes from `Output/transcripts/<name>.lang` (written by `transcribe.py` /
  `youtube_transcript.py`), or is auto-detected from the transcript when that file is missing.
  Do not force `--language english` on a Spanish recording.
- `summarize.py` verifies every summary with a dependency-free detector
  (`detect_language`, `summary_language_ok`). On mismatch it regenerates once; if it still drifts it
  saves the result but logs `status=lang-mismatch` in `Docs/operations-log.csv`.
- Report the language check result alongside the summary stats. If a mismatch survives the retry,
  say it plainly instead of shipping the file as fine.

## Where outputs go

`Output/` is a **staging area, not the destination.** Every run's artifacts belong in a downstream
knowledge repo, filed by topic:

- **Tax-preparer recordings → `ky-tax-preparer`** (dedicated repo; `exploring-pathways` only keeps a
  pointer README for that pathway):

  ```
  ky-tax-preparer/course-archives/<Brand>/<ClaseN | YYYYMMDD>/
      <stem>.txt
      <stem>.lang
      <stem>_timestamps.txt
      <stem>_summary_openai_luna.md
  ```

- **Every other topic → `exploring-pathways`:**

  ```
  exploring-pathways/Pathways/<Topic>/<Brand or Person>/<ClaseN | YYYYMMDD>/
  ```

After a run, move the artifacts there (transcript, `.lang`, timestamps and the summary), and
leave `Output/` empty apart from `.gitkeep`:

```bash
uv run scripts/Python/file_outputs.py "<stem>" --dest "C:\\...\\ky-tax-preparer\\course-archives\\<Brand>\\<Clase>"
uv run scripts/Python/file_outputs.py "<stem>" --dest "C:\\...\\exploring-pathways\\Pathways\\<Topic>\\<Brand>\\<Clase|Date>"
```

The helper moves by default (use `--keep-source` to copy, `--dry-run` to preview) and reports
missing artifacts instead of failing. Topic folders in use:

| Recording type | Folder |
|---|---|
| Tax preparer courses | `ky-tax-preparer\course-archives\<Brand>\<ClaseN>` (e.g. `HugoTaxPro\Preparatorio`, `HugoTaxPro\Clase3`, `TaxMaster\Clase3`, `ClasesDeTaxes[\Software]`, `EdukInstitute\Clase3`, `KTA_Institute\Classes\<YYYYMMDD>`, `ProFinanceAcademy\Classes\<YYYYMMDD>`) |
| Credit repair / funding | `exploring-pathways\Pathways\CreditRepair\<Brand>[\<Person>][\<YYYYMMDD>]` (e.g. `UnionSolutions\20261020`, `CashFlowVIP\20261003`, `CapitalFunding\JoseKalil`) |
| Cybersecurity | `exploring-pathways\Pathways\Cyber\<Vendor>\<Person>` or `CyberSec\<Program>` |
| Insurance sales | `exploring-pathways\Pathways\Insurance\<Person>` (e.g. `Insurance\GustavoAbreu`) |
| TikTok Shop | `exploring-pathways\Pathways\TikTok\TikTokWiz\<YYYYMMDD>` |
| Machine learning | `exploring-pathways\Pathways\ML\<Person>\<YYYYMMDD>` |
| Digital marketing | `exploring-pathways\Pathways\DigitalMarketing\<Person>\<YYYYMMDD>` |
| Amazon / e-commerce | `exploring-pathways\Pathways\Amazon\<Brand>\<YYYYMMDD>` |

When a topic is new, follow the existing pattern (topic → brand/person → class or date) instead of
creating a flat folder. Those artifacts are the user's working documents, so don't delete or
reorganize archived files without being asked.

## Downloading audio (MP3) from a URL

`scripts/Python/download_audio.py` is a standalone downloader (yt-dlp + ffmpeg + Deno), separate
from the summarize pipeline:

```bash
uv run scripts/Python/download_audio.py "<url>" [--preset mp3|mp3-vbr|m4a|opus|flac|wav|native]
```

- Output goes to `Output/downloads/<preset>/` (git-ignored). That is a **personal media
  library**, not a knowledge-repo destination — do not file those MP3s into `exploring-pathways`
  or `ky-tax-preparer`. It is also **exempt from the "leave `Output/` empty" rule** above: never
  move or delete `Output/downloads/` when cleaning up after a transcript/summary run.
- Default is `m4a`, a bit-exact stream copy of YouTube's AAC audio (plays on Windows/macOS/iOS/
  Android and modern car stereos). `--preset mp3` re-encodes the already-lossy source: bigger and
  slightly worse, only for MP3-only hardware. YouTube has no 320 kbps stream (~128-130 kbps AAC is
  the ceiling), so never present MP3 320 as an upgrade in quality.
- `--dry-run` previews titles/durations, `--playlist` expands playlists, `--batch-file urls.txt`
  queues many URLs, `--cookies-from-browser chrome` handles age-restricted/bot-checked videos.
- The script detects `deno`/`node`/`bun` and passes them to yt-dlp (which only enables
  `deno` by itself), printing an install hint when none is found. If a user reports "fewer
  formats"/low quality on a new machine, that warning is the first thing to look for.

## Standard workflow for a local audio/video file

1. Verify the file exists and get its duration with `ffprobe`.
2. Transcribe:
   `uv run python -u scripts/Python/transcribe.py "<path>" --model medium`
   → `Output/transcripts/<name>.txt` (faster-whisper `medium`, CUDA FP16 on this machine).
3. Summarize with Luna:
   `uv run scripts/Python/summarize.py "Output/transcripts/<name>.txt" --provider openai --openai-model gpt-6-luna`
   → `Output/<name>_summary_openai_luna.md` (the tag comes from the model, e.g. `_summary_openai_terra.md`).
4. Validate: check the file exists and is non-trivial (chars/words/lines), then read the first
   ~20 lines to confirm the content matches the recording.
5. File the artifacts into the topic folder in `exploring-pathways` with
   `scripts/Python/file_outputs.py` (see *Where outputs go* above).
6. Report: source, duration, detected language, transcript size, summary path + size, the
   language-check result, and a short topic summary. Mention any anomaly (mis-detected language,
   truncated audio, retries).

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
