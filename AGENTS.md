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

`Output/` is a **staging area, not the destination.** Every run's artifacts belong in the
downstream knowledge repo, filed by topic:

```
exploring-pathways/Pathways/<Topic>/<Brand or Person>/<ClaseN | YYYYMMDD>/
    <stem>.txt
    <stem>.lang
    <stem>_timestamps.txt
    <stem>_summary_luna.md
```

After a run, move the artifacts there (transcript, `.lang`, timestamps and the summary), and
leave `Output/` empty apart from `.gitkeep`:

```bash
uv run scripts/Python/file_outputs.py "<stem>" --dest "C:\\...\\exploring-pathways\\Pathways\\<Topic>\\<Brand>\\<Clase|Date>"
```

The helper moves by default (use `--keep-source` to copy, `--dry-run` to preview) and reports
missing artifacts instead of failing. Topic folders in use:

| Recording type | Folder |
|---|---|
| Tax preparer courses | `TaxPreparer\<Brand>\<ClaseN>` (e.g. `TaxMaster\Clase2`, `HugoTaxPro\Preparatorio`, `ClasesDeTaxes[\Software]`, `EdukInstitute\Clase3`, `KTA_Institute\Classes`, `ProFinanceAcademy\Classes`) |
| Credit repair / funding | `CreditRepair\<Brand>[\<Person>][\<YYYYMMDD>]` (e.g. `UnionSolutions\20261020`, `CashFlowVIP\20261003`, `CapitalFunding\JoseKalil`) |
| Cybersecurity | `Cyber\<Vendor>\<Person>` or `CyberSec\<Program>` |
| Insurance sales | `Insurance\<Person>` (e.g. `Insurance\GustavoAbreu`) |
| TikTok Shop | `TikTok\TikTokWiz\<YYYYMMDD>` |
| Machine learning | `ML\<Person>\<YYYYMMDD>` |
| Digital marketing | `DigitalMarketing\<Person>\<YYYYMMDD>` |
| Amazon / e-commerce | `Amazon\<Brand>\<YYYYMMDD>` |

When a topic is new, follow the existing pattern (topic → brand/person → class or date) instead of
creating a flat folder. Those artifacts are the user's working documents, so don't delete or
reorganize archived files without being asked.

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
