"""Download audio from YouTube (or any other yt-dlp supported site) as M4A/MP3/Opus.

This is the 4KVideoDownloader-style tool: give it one or more URLs and it saves
audio files with proper metadata and embedded cover art.

Quality notes (important):
  * YouTube's best audio stream is usually Opus ~106-160 kbps (webm) or AAC
    ~128-130 kbps (m4a). Those are the *source*; there is no 320 kbps to be had.
  * The default preset is `m4a`, which stream-copies YouTube's AAC audio: the
    file is bit-for-bit the same audio, just with tags and cover art added.
  * MP3 320 kbps cannot restore detail that was never there — it re-encodes the
    already-lossy source, so it is bigger *and* slightly worse. Choose it only
    for maximum device compatibility (old car stereos, MP3-only players, DJ
    software that refuses AAC). `mp3-vbr` (LAME V0) is the best MP3 option.
  * `opus` copies YouTube's Opus stream (smallest, highest fidelity per byte);
    `native` keeps the original container with no conversion at all.

Multi-language audio and broken tracks:
  * Videos with YouTube auto-dubbing expose several audio tracks (e.g. `140-0`
    English, `140-2` Spanish, `140-7` original). yt-dlp prefers the track marked
    "(default)", which is normally the original — but YouTube sometimes serves a
    broken one (a few seconds of audio, then silence, repeating).
  * You do not need to do anything about it: every download is scanned for
    silence, and a mostly-silent track is automatically replaced by the first
    clean alternative track (requested language first, then Spanish/English).
    Silence at the very start or end of a video is reported as a note, since no
    track can fix an upload that is itself padded with silence.
  * `--trim-silence` additionally cuts silence (5 s or longer) off the end of
    each file with a stream copy, keeping 0.5 s after the last sound.
  * `--audio-lang es` only expresses a preference for videos that really are
    multi-language (e.g. a talk with real dubs); it is never required.

Usage:
    uv run scripts/Python/download_audio.py "https://youtu.be/VIDEO_ID"
    uv run scripts/Python/download_audio.py URL1 URL2 URL3 --preset mp3
    uv run scripts/Python/download_audio.py "https://www.youtube.com/playlist?list=..." --playlist
    uv run scripts/Python/download_audio.py --batch-file urls.txt --preset mp3-vbr
    uv run scripts/Python/download_audio.py URL --dry-run
    uv run scripts/Python/download_audio.py URL --audio-lang es

Output defaults to Output/downloads/<preset>/ (git-ignored).
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.cookies import SUPPORTED_BROWSERS
from yt_dlp.utils import DownloadCancelled, ExistingVideoReached

# Windows consoles default to cp1252, which crashes on emoji/CJK in video
# titles (same fix as transcribe.py / youtube_transcript.py).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_OUTPUT_DIR = Path("Output/downloads")


@dataclass(frozen=True)
class Preset:
    """A target audio format plus the yt-dlp postprocessors that produce it."""

    key: str
    description: str
    # Which source stream to fetch. Prefer the codec we can stream-copy.
    format_selector: str
    postprocessors: tuple[dict, ...]
    # Expected container extension, or None when the source container is kept.
    extension: str | None
    # True when audio is re-encoded (i.e. quality is bounded by the source).
    reencodes: bool


PRESETS: dict[str, Preset] = {
    "m4a": Preset(
        key="m4a",
        description="M4A/AAC - bit-exact stream copy of YouTube's AAC audio (default)",
        format_selector="bestaudio[ext=m4a]/bestaudio[acodec^=mp4a]/bestaudio/best",
        postprocessors=(
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "2"},
        ),
        extension="m4a",
        reencodes=False,
    ),
    "mp3": Preset(
        key="mp3",
        description="MP3 320 kbps CBR - maximum compatibility, but re-encodes (lossy)",
        format_selector="bestaudio/best",
        postprocessors=(
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "320"},
        ),
        extension="mp3",
        reencodes=True,
    ),
    "mp3-vbr": Preset(
        key="mp3-vbr",
        description="MP3 V0 VBR (~245 kbps) - best quality MP3 per byte",
        format_selector="bestaudio/best",
        postprocessors=(
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "0"},
        ),
        extension="mp3",
        reencodes=True,
    ),
    "opus": Preset(
        key="opus",
        description="Opus - true stream copy of YouTube's Opus stream (highest fidelity per byte)",
        # Opus only: yt-dlp's FFmpegExtractAudio re-encodes Opus->Opus because it
        # maps the target to libopus, so remux the container instead (-c copy).
        format_selector="bestaudio[acodec=opus]/bestaudio[ext=webm]",
        postprocessors=(
            # yt-dlp's remux PP takes the (misspelled) 'preferedformat' kwarg.
            {"key": "FFmpegVideoRemuxer", "preferedformat": "opus"},
        ),
        extension="opus",
        reencodes=False,
    ),
    "flac": Preset(
        key="flac",
        description="FLAC - lossless container for the decoded source (large, no gain over source)",
        format_selector="bestaudio/best",
        postprocessors=(
            {"key": "FFmpegExtractAudio", "preferredcodec": "flac", "preferredquality": "0"},
        ),
        extension="flac",
        reencodes=True,
    ),
    "wav": Preset(
        key="wav",
        description="WAV 16-bit PCM - uncompressed, for editing",
        format_selector="bestaudio/best",
        postprocessors=(
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"},
        ),
        extension="wav",
        reencodes=True,
    ),
    "native": Preset(
        key="native",
        description="Keep the original stream untouched (no conversion, no quality loss)",
        format_selector="bestaudio/best",
        postprocessors=(),
        extension=None,
        reencodes=False,
    ),
}


def check_ffmpeg() -> bool:
    """ffmpeg is required for conversion/remuxing and for embedding tags/art."""
    if shutil.which("ffmpeg"):
        return True
    print("Error: ffmpeg not found on PATH. Install it with: winget install --id Gyan.FFmpeg")
    return False


JS_RUNTIMES = ("deno", "node", "bun")

# yt-dlp colorizes the strings it hands to progress hooks (e.g. '\x1b[0;33m00:00\x1b[0m').
# Printing those verbatim is what used to leave the terminal tinted yellow when a
# line got truncated mid-sequence, so strip them and print plain text.
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return ANSI_RE.sub("", str(text))


def fit_to_width(text: str, width: int) -> str:
    """Truncate and pad plain text to exactly `width` characters."""
    return text[:width].ljust(width)


def browser_name(value: str) -> str:
    """Accept `Chrome`, `CHROME`, ... — yt-dlp itself only takes lowercase names."""
    name = value.strip().lower()
    if name not in SUPPORTED_BROWSERS:
        raise argparse.ArgumentTypeError(
            f"unsupported browser '{value}' (choose from: {', '.join(sorted(SUPPORTED_BROWSERS))})"
        )
    return name


BOT_CHECK_MARKERS = ("confirm you’re not a bot", "confirm you're not a bot")


def failure_hint(message: str, used_cookies: bool) -> str | None:
    """Plain-language next step for the errors users actually hit."""
    if "cookie database" in message:
        return (
            "The browser has its cookie file locked. Quit it completely first (Chrome and Edge\n"
            "  keep running in the background: right-click their tray icon > Exit, or run\n"
            "  `taskkill /F /IM chrome.exe`), then retry. Or export cookies.txt with a browser\n"
            "  extension ('Get cookies.txt LOCALLY') and use --cookies path\\to\\cookies.txt."
        )
    if not any(marker in message for marker in BOT_CHECK_MARKERS):
        return None
    if used_cookies:
        return (
            "YouTube still wants a sign-in. Make sure you are logged in to YouTube in that\n"
            "  browser and CLOSE the browser first (Chrome/Edge lock their cookie file);\n"
            "  if Chrome or Edge keep failing, try --cookies-from-browser firefox."
        )
    return (
        "YouTube is asking for a sign-in check (usually a temporary block on your IP).\n"
        "  Retry with your browser's cookies, e.g.:\n"
        "      uv run scripts/Python/download_audio.py URL --cookies-from-browser chrome\n"
        "  or just wait a while and try again."
    )


def with_language_filter(selector: str, language: str) -> str:
    """Restrict every alternative of a format selector to one audio language.

    `bestaudio[ext=m4a]/best` becomes
    `bestaudio[ext=m4a][language^=es]/best[language^=es]/bestaudio[ext=m4a]/best`,
    so multi-track videos (YouTube auto-dubbing) resolve to the requested
    language instead of the track marked "(default)". The prefix match covers
    regional codes such as `es-MX` when `es` is requested. Most videos carry no
    language tag at all, so the unfiltered selector stays as a fallback.
    """
    constraint = f"[language^={language}]"
    alternatives = []
    for alternative in selector.split("/"):
        alternative = alternative.strip()
        if not alternative:
            continue
        if "]" in alternative:
            head, _, _ = alternative.rpartition("]")
            alternatives.append(f"{head}]{constraint}")
        else:
            alternatives.append(f"{alternative}{constraint}")
    return "/".join(alternatives + [selector])


# Broken source tracks keep a continuous timeline and just carry empty samples,
# so the defect is long silence runs rather than missing packets. Where the
# silence sits matters:
#   * short pauses between songs        -> normal, ignored
#   * silence at the very start or end  -> the upload itself is padded; no other
#                                          track or download setting can fix it
#   * lots of silence in the middle     -> a broken track; try another one
SILENCE_NOISE = "-50dB"
SILENCE_MIN_SECONDS = 2.0
SILENCE_SHARE_THRESHOLD = 0.15
SILENCE_MIN_DEFECT_SECONDS = 10.0
PADDING_NOTE_SECONDS = 30.0
MAX_TRACK_RETRIES = 3
TRIM_MIN_SECONDS = 5.0
TRIM_KEEP_SECONDS = 0.5

SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")


@dataclass
class SilenceReport:
    """Silent runs of a file, split by where they sit."""

    duration: float
    inner: list[tuple[float, float]] = field(default_factory=list)
    leading: float = 0.0
    trailing: float = 0.0
    trailing_start: float | None = None

    @property
    def inner_seconds(self) -> float:
        return sum(end - start for start, end in self.inner)

    @property
    def inner_share(self) -> float:
        return self.inner_seconds / self.duration if self.duration else 0.0

    @property
    def defective(self) -> bool:
        return (
            self.inner_seconds >= SILENCE_MIN_DEFECT_SECONDS
            and self.inner_share >= SILENCE_SHARE_THRESHOLD
        )


def silence_report(path: Path) -> SilenceReport | None:
    """Scan a finished file for silent runs of at least ~2 s; None if it can't be scanned."""
    if not shutil.which("ffmpeg"):
        return None
    duration = float((probe_audio(path).get("format") or {}).get("duration") or 0)
    if not duration:
        return None
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-nostats", "-nostdin", "-v", "info",
                "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                "-af", f"silencedetect=noise={SILENCE_NOISE}:d={SILENCE_MIN_SECONDS:g}",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600, check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None

    runs: list[tuple[float, float]] = []
    pending: float | None = None
    for line in completed.stderr.splitlines():
        started = SILENCE_START_RE.search(line)
        if started:
            pending = max(float(started.group(1)), 0.0)
            continue
        ended = SILENCE_END_RE.search(line)
        if ended and pending is not None:
            runs.append((pending, float(ended.group(1))))
            pending = None
    if pending is not None:
        runs.append((pending, duration))

    report = SilenceReport(duration=duration)
    edge = 1.0
    for start, end in runs:
        at_start = start <= edge
        at_end = end >= duration - edge
        if at_end and not at_start:
            report.trailing += end - start
            if report.trailing_start is None or start < report.trailing_start:
                report.trailing_start = start
        elif at_start and not at_end:
            report.leading += end - start
        else:
            report.inner.append((start, end))
    return report


def silence_notes(report: SilenceReport) -> list[str]:
    """Human-readable lines for padding at the start/end of a file."""
    notes = []
    if report.trailing >= PADDING_NOTE_SECONDS:
        notes.append(
            f"  Note: the last {human_duration(report.trailing)} of this file is silent. "
            "The video itself is silent there; no download option changes that."
        )
    if report.leading >= PADDING_NOTE_SECONDS:
        notes.append(
            f"  Note: the first {human_duration(report.leading)} of this file is silent. "
            "The video itself is silent there; no download option changes that."
        )
    return notes


def alternative_audio_tracks(info: dict, preferred_language: str | None) -> list[str]:
    """Other audio tracks of the same stream (e.g. `140-2` for `140-7`), best first.

    YouTube auto-dubbing publishes one stream per language under the same base
    id. For music they carry the same audio, so any clean one will do; the
    requested language (default Spanish, then English) is tried first.
    """
    chosen = ((info.get("requested_downloads") or [{}])[0]).get("format_id")
    chosen = str(chosen or info.get("format_id") or "")
    if "-" not in chosen:
        return []
    base = chosen.split("-")[0]
    want = (preferred_language or "es").lower()

    def rank(language: str) -> tuple[bool, bool]:
        language = language.lower()
        return (not language.startswith(want), not language.startswith("en"))

    candidates = []
    for fmt in info.get("formats") or []:
        format_id = str(fmt.get("format_id") or "")
        if format_id == chosen or "-" not in format_id or format_id.split("-")[0] != base:
            continue
        candidates.append((rank(str(fmt.get("language") or "")), format_id))
    return [format_id for _, format_id in sorted(candidates)]


def detect_js_runtimes() -> dict[str, dict]:
    """Return a yt-dlp `js_runtimes` config for every runtime found on PATH.

    yt-dlp only enables `deno` on its own, so a machine that has Node or Bun
    installed would still fall back to "no JS runtime" (and silently fewer
    formats). Enabling what we find makes any of them work.
    """
    return {runtime: {} for runtime in JS_RUNTIMES if shutil.which(runtime)}


def warn_if_no_js_runtime(urls: list[str], runtimes: dict[str, dict]) -> None:
    """Warn about a missing JS runtime instead of letting yt-dlp degrade silently.

    yt-dlp needs a JavaScript engine to solve YouTube's signature/`n` parameters.
    Without one it still downloads, but prints a quiet warning and offers fewer
    formats — so on a fresh machine the result looks "fine but worse" rather
    than broken.
    """
    if runtimes or not any("youtube.com" in u or "youtu.be" in u for u in urls):
        return
    print("Warning: no JavaScript runtime (deno/node/bun) found on PATH.")
    print("         YouTube extraction will still work, but may miss formats/quality.")
    print("         Fix: winget install --id DenoLand.Deno  (then restart the shell)")


def human_size(num_bytes: float | None) -> str:
    if not num_bytes:
        return "?"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return "?"


def human_duration(seconds: float | None) -> str:
    if not seconds:
        return "?"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@dataclass
class ProgressPrinter:
    """Single-line live progress that does not spam the console."""

    quiet: bool = False
    _last_update: float = field(default=0.0, repr=False)

    def __call__(self, event: dict) -> None:
        if self.quiet:
            return

        status = event.get("status")
        if status == "downloading":
            now = time.monotonic()
            if now - self._last_update < 0.4:
                return
            self._last_update = now
            total = (
                event.get("_total_bytes_str")
                or event.get("_total_bytes_estimate_str")
                or "?"
            )
            line = (
                f"  {event.get('_percent_str', '').strip()} of {total}"
                f"  {event.get('_speed_str', '?')}  ETA {event.get('_eta_str', '?')}"
            )
            sys.stdout.write("\r" + fit_to_width(strip_ansi(line), self._width()))
            sys.stdout.flush()
        elif status == "finished":
            self._clear()
            print(f"  Downloaded {human_size(event.get('total_bytes'))}, processing...")

    @staticmethod
    def _width() -> int:
        return max(shutil.get_terminal_size((80, 24)).columns - 1, 20)

    def _clear(self) -> None:
        if not self.quiet:
            sys.stdout.write("\r" + " " * self._width() + "\r")
            sys.stdout.flush()

    def newline(self) -> None:
        self._clear()


def build_options(
    preset: Preset,
    output_dir: Path,
    progress: ProgressPrinter,
    embed_metadata: bool,
    embed_thumbnail: bool,
    archive_file: Path | None,
    cookies_from_browser: str | None,
    playlist: bool,
    quiet: bool,
    js_runtimes: dict[str, dict] | None = None,
    audio_language: str | None = None,
    format_override: str | None = None,
    cookies_file: str | None = None,
) -> dict:
    """Assemble the yt-dlp option dict for one preset.

    `format_override` pins one exact format id (used to retry another audio track).
    """
    outtmpl = str(output_dir / "%(title)s [%(id)s].%(ext)s")
    selector = preset.format_selector
    if audio_language:
        selector = with_language_filter(selector, audio_language)
    if format_override:
        selector = format_override

    postprocessors: list[dict] = list(preset.postprocessors)
    if embed_metadata:
        postprocessors.append({"key": "FFmpegMetadata", "add_metadata": True})
    if embed_thumbnail and preset.key not in ("wav", "flac", "native"):
        # MP3/M4A/Opus support attached cover art; WAV/FLAC need extra tagging
        # support, so leave them alone.
        postprocessors.append({"key": "EmbedThumbnail", "already_have_thumbnail": False})

    opts: dict = {
        "format": selector,
        "outtmpl": outtmpl,
        "noplaylist": not playlist,
        # In a playlist, one unavailable/private video must not abort the rest.
        "ignoreerrors": "only_download" if playlist else False,
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 4,
        "writethumbnail": embed_thumbnail and preset.key not in ("wav", "flac", "native"),
        "postprocessors": postprocessors,
        "progress_hooks": [progress],
        "quiet": True,
        "no_warnings": quiet,
        "noprogress": True,
        "consoletitle": False,
        # Print our own plain-text progress; colored output from yt-dlp would
        # otherwise leak escape codes into captured logs and truncate badly.
        "color": "no_color",
    }

    if archive_file is not None:
        opts["download_archive"] = str(archive_file)
    if cookies_file:
        opts["cookiefile"] = str(cookies_file)
    elif cookies_from_browser:
        # Needed for age-restricted videos and "confirm you're not a bot" walls.
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if js_runtimes:
        # yt-dlp defaults to deno only; pass whatever is actually installed.
        opts["js_runtimes"] = js_runtimes

    return opts


def probe_audio(path: Path) -> dict:
    """Read duration/bitrate/codec back from the finished file."""
    if not shutil.which("ffprobe"):
        return {}
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_streams", "-show_format",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60, check=False,
        )
        if completed.returncode != 0:
            return {}
        return json.loads(completed.stdout or "{}")
    except Exception:
        return {}


def _attached_picture_bytes(path: Path) -> int:
    """Total size of cover-art packets, so they don't inflate the audio bitrate."""
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v",
                "-show_entries", "packet=size", "-of", "csv=p=0", str(path),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60, check=False,
        )
        return sum(int(line) for line in completed.stdout.split() if line.isdigit())
    except Exception:
        return 0


def describe_file(path: Path) -> str:
    data = probe_audio(path)
    streams = data.get("streams") or []
    # Report the audio stream only: embedded cover art lives in a separate
    # video stream and would otherwise inflate the container bitrate.
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    fmt = data.get("format") or {}

    duration = float(fmt.get("duration") or audio.get("duration") or 0) or None
    bitrate = audio.get("bit_rate")
    if not bitrate and duration:
        # Ogg/Opus and friends do not report per-stream bitrates; derive it from
        # the file size minus the cover art.
        audio_bytes = max(path.stat().st_size - _attached_picture_bytes(path), 0)
        bitrate = audio_bytes * 8 / duration
    kbps = f"{int(int(bitrate) / 1000)} kbps" if bitrate else "?"
    codec = audio.get("codec_name") or "?"
    rate = audio.get("sample_rate")
    channels = audio.get("channels")

    parts = [
        human_size(path.stat().st_size),
        human_duration(duration),
        kbps,
        codec.upper() if codec != "?" else "?",
    ]
    if rate:
        parts.append(f"{int(rate) / 1000:g} kHz")
    if channels:
        parts.append(f"{channels}ch")
    return " / ".join(parts)


def split_preset_from_output(preset_key: str, output_dir: Path) -> Path:
    """Give each preset its own subfolder so MP3 and Opus never overwrite each other."""
    return output_dir / preset_key


def dry_run(urls: list[str], preset: Preset, playlist: bool, quiet: bool) -> int:
    js_runtimes = detect_js_runtimes()
    warn_if_no_js_runtime(urls, js_runtimes)
    print(f"\nDry run - preset '{preset.key}': {preset.description}")
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": not playlist,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "js_runtimes": js_runtimes or None,
    }
    failures = 0
    with YoutubeDL(opts) as ydl:
        for url in urls:
            try:
                info = ydl.extract_info(url, download=False)
            except Exception as exc:
                print(f"  FAILED {url}: {exc}")
                failures += 1
                continue
            if not info:
                print(f"  SKIP   {url} (no info returned)")
                failures += 1
                continue

            entries = info.get("entries")
            if entries:
                entries = [e for e in entries if e]
                total = sum((e.get("duration") or 0) for e in entries)
                print(f"  PLAYLIST {info.get('title')} - {len(entries)} items, "
                      f"{human_duration(total)} total")
                if not quiet:
                    for entry in entries[:50]:
                        print(f"    - {entry.get('title')} ({human_duration(entry.get('duration'))})")
                    if len(entries) > 50:
                        print(f"    ... and {len(entries) - 50} more")
            else:
                print(f"  OK     {info.get('title')} "
                      f"({human_duration(info.get('duration'))}, "
                      f"{info.get('uploader') or '?'})")
    return 1 if failures else 0


@dataclass
class DownloadSummary:
    """Outcome of a download run."""

    results: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def download_audio(
    urls: list[str],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    preset_key: str = "m4a",
    embed_metadata: bool = True,
    embed_thumbnail: bool = True,
    use_archive: bool = True,
    cookies_from_browser: str | None = None,
    playlist: bool = False,
    quiet: bool = False,
    audio_language: str | None = None,
    trim_silence: bool = False,
    cookies_file: str | None = None,
) -> DownloadSummary:
    """Download audio for every URL and report what landed on disk.

    Files are named `<title> [<video_id>].<ext>` and placed in
    `<output_dir>/<preset>/`.
    """
    preset = PRESETS[preset_key]
    if not check_ffmpeg():
        return DownloadSummary(failures=[("<setup>", "ffmpeg not available")])

    js_runtimes = detect_js_runtimes()
    warn_if_no_js_runtime(urls, js_runtimes)

    target_dir = split_preset_from_output(preset.key, Path(output_dir))
    target_dir.mkdir(parents=True, exist_ok=True)

    archive_file = target_dir / ".downloaded.txt" if use_archive else None
    progress = ProgressPrinter(quiet=quiet)

    opts = build_options(
        preset=preset,
        output_dir=target_dir,
        progress=progress,
        embed_metadata=embed_metadata,
        embed_thumbnail=embed_thumbnail,
        archive_file=archive_file,
        cookies_from_browser=cookies_from_browser,
        playlist=playlist,
        quiet=quiet,
        js_runtimes=js_runtimes,
        audio_language=audio_language,
        cookies_file=cookies_file,
    )

    if not quiet:
        print(f"Preset : {preset.key} - {preset.description}")
        print(f"Output : {target_dir.resolve()}")
        if audio_language:
            print(f"Audio language: {audio_language}")
        if preset.reencodes:
            print("Note   : re-encoding from the source stream; quality is capped by YouTube's audio.")

    def make_options(retry_dir: Path, format_id: str) -> dict:
        return build_options(
            preset=preset,
            output_dir=retry_dir,
            progress=ProgressPrinter(quiet=True),
            embed_metadata=embed_metadata,
            embed_thumbnail=embed_thumbnail,
            archive_file=None,
            cookies_from_browser=cookies_from_browser,
            playlist=False,
            quiet=True,
            js_runtimes=js_runtimes,
            format_override=format_id,
            cookies_file=cookies_file,
        )

    results: list[Path] = []
    notes: dict[Path, list[str]] = {}
    skipped: list[str] = []
    failures: list[tuple[str, str]] = []
    started = time.monotonic()

    with YoutubeDL(opts) as ydl:
        for index, url in enumerate(urls, start=1):
            if not quiet:
                print(f"\n[{index}/{len(urls)}] {url}")
            try:
                info = ydl.extract_info(url, download=True)
            except (ExistingVideoReached, DownloadCancelled):
                progress.newline()
                skipped.append(url)
                print("  Already in the download archive, skipped.")
                continue
            except Exception as exc:
                message = str(exc).removeprefix("ERROR: ")
                failures.append((url, message))
                progress.newline()
                print(f"  FAILED: {message}")
                continue

            if info is None:
                # yt-dlp returns None (instead of raising) when the archive
                # already contains every requested video.
                progress.newline()
                skipped.append(url)
                print("  Already in the download archive, skipped.")
                continue

            raw_entries = info.get("entries")
            entries = [e for e in raw_entries or [] if e]
            if raw_entries is not None:
                missed = len(raw_entries) - len(entries)
                if missed:
                    failures.append((url, f"{missed} playlist item(s) failed (see errors above)"))
            requests = entries or ([] if raw_entries is not None else [info])
            for request in requests:
                path = resolve_output_path(request, target_dir, preset)
                if path is not None:
                    results.append(path)
                    notes[path] = verify_audio(
                        path, request, url, preset, target_dir, make_options, audio_language,
                        trim_silence,
                    )

    elapsed = time.monotonic() - started

    if results:
        print("\n=== Finished ===")
        for path in results:
            print(f"  {path.name}")
            print(f"    {describe_file(path)}")
            print(f"    {path}")
            for note in notes.get(path, []):
                print(note)
    if skipped:
        print(f"\n{len(skipped)} input(s) already downloaded (use --no-archive to fetch again).")
    if failures:
        print(f"\n{len(failures)} of {len(urls)} input(s) failed:")
        for url, message in failures:
            print(f"  {url}\n    {message}")
        hints = {
            hint for _, message in failures
            if (hint := failure_hint(message, bool(cookies_from_browser or cookies_file)))
        }
        for hint in sorted(hints):
            print(f"\n  Hint: {hint}")

    total_bytes = sum(p.stat().st_size for p in results if p.exists())
    print(f"\n{len(results)} file(s), {human_size(total_bytes)} in {elapsed:.1f}s")
    return DownloadSummary(results=results, skipped=skipped, failures=failures)


def trim_trailing_silence(path: Path, report: SilenceReport) -> list[str]:
    """Cut silence off the end of a file with a stream copy (no re-encode).

    Keeps half a second after the last sound so the fade-out is not clipped.
    Only the end is trimmed: seeking past the start in copy mode can drop the
    embedded cover art. On any failure the original file is left untouched.
    """
    if report.trailing < TRIM_MIN_SECONDS or report.trailing_start is None:
        return []
    keep = min(report.trailing_start + TRIM_KEEP_SECONDS, report.duration)
    temp = path.with_name(f".trim{path.suffix}")
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(path),
                "-t", f"{keep:.3f}", "-map", "0", "-c", "copy", "-map_metadata", "0",
                str(temp),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600, check=False,
        )
        new_duration = float((probe_audio(temp).get("format") or {}).get("duration") or 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        temp.unlink(missing_ok=True)
        return ["  Note: could not trim the trailing silence; the file was left as downloaded."]

    # A good cut is shorter than the original and close to the requested length.
    if completed.returncode != 0 or not (0 < new_duration < report.duration - 1.0) \
            or abs(new_duration - keep) > 2.0:
        temp.unlink(missing_ok=True)
        return ["  Note: could not trim the trailing silence; the file was left as downloaded."]

    temp.replace(path)
    removed = report.duration - new_duration
    report.duration, report.trailing, report.trailing_start = new_duration, 0.0, None
    return [f"  Trimmed {human_duration(removed)} of trailing silence "
            f"({human_duration(new_duration + removed)} -> {human_duration(new_duration)})."]


def verify_audio(
    path: Path,
    info: dict,
    url: str,
    preset: Preset,
    target_dir: Path,
    make_options,
    audio_language: str | None,
    trim_silence: bool = False,
) -> list[str]:
    """Check a finished file for a broken (mostly silent) track and fix it if possible.

    When the chosen track is defective, the other audio tracks of the same video
    are tried in turn; the cleanest one replaces the file. Returns the notes and
    warnings to print under the file.
    """
    report = silence_report(path)
    if report is None:
        return []

    def finish(final_path: Path, final: SilenceReport, notes: list[str]) -> list[str]:
        if trim_silence:
            notes = notes + trim_trailing_silence(final_path, final)
        return notes + silence_notes(final)

    if not report.defective:
        return finish(path, report, [])

    notes: list[str] = []
    best_path, best = path, report
    tracks = alternative_audio_tracks(info, audio_language)[:MAX_TRACK_RETRIES]
    if tracks:
        print(f"  {report.inner_share * 100:.0f}% of the audio is silent - "
              f"trying {len(tracks)} other audio track(s)...")

    retry_dir = target_dir / ".retry"
    source_url = info.get("webpage_url") or info.get("original_url") or url
    switched_to = None
    for format_id in tracks:
        shutil.rmtree(retry_dir, ignore_errors=True)
        retry_dir.mkdir(parents=True, exist_ok=True)
        try:
            with YoutubeDL(make_options(retry_dir, format_id)) as ydl:
                retry_info = ydl.extract_info(source_url, download=True)
        except Exception as exc:  # noqa: BLE001 - any failure just means "try the next track"
            print(f"    track {format_id}: failed ({str(exc).splitlines()[0][:80]})")
            continue
        candidate = resolve_output_path(retry_info or {}, retry_dir, preset)
        candidate_report = silence_report(candidate) if candidate else None
        if candidate is None or candidate_report is None:
            continue
        print(f"    track {format_id}: {candidate_report.inner_share * 100:.0f}% silent")
        if candidate_report.inner_share < best.inner_share:
            if best_path != path:
                best_path.unlink(missing_ok=True)
            kept = retry_dir.with_name(f".keep{candidate.suffix}")
            candidate.replace(kept)
            best_path, best, switched_to = kept, candidate_report, format_id
        if not candidate_report.defective:
            break
    shutil.rmtree(retry_dir, ignore_errors=True)

    final_path = path
    if best_path != path:
        final_path = path.with_suffix(best_path.suffix)
        if final_path != path:
            path.unlink(missing_ok=True)
        best_path.replace(final_path)
        notes.append(f"  Note: the default audio track was broken; used track {switched_to} instead.")

    if best.defective:
        notes.append(
            f"  WARNING: {human_duration(best.inner_seconds)} of this file "
            f"({best.inner_share * 100:.0f}%) is silent, spread over {len(best.inner)} stretches.\n"
            "           The source audio is damaged and no other track of this video was clean."
        )
    return finish(final_path, best, notes)


def resolve_output_path(info: dict, output_dir: Path, preset: Preset) -> Path | None:
    """Locate the finished file described by a yt-dlp info dict."""
    candidates: list[str] = []

    requested = (info.get("requested_downloads") or [{}])[0]
    for key in ("filepath", "_filename", "filename"):
        value = requested.get(key) or info.get(key)
        if value:
            candidates.append(str(value))

    for raw in candidates:
        path = Path(raw)
        stem = path.with_suffix("")
        if preset.extension:
            converted = stem.with_suffix("." + preset.extension)
            if converted.exists():
                return converted
        if path.exists():
            return path
        # Some postprocessors rewrite the name entirely (e.g. "Title [id].mp3").
        for match in stem.parent.glob(stem.name + ".*"):
            if match.is_file() and match.suffix.lower() != ".part":
                return match

    # Last resort: anything in the download dir matching this video id.
    video_id = info.get("id")
    if video_id:
        found = sorted(
            p for p in output_dir.glob(f"*{video_id}*")
            if p.is_file() and p.suffix.lower() not in (".part", ".ytdl", ".jpg", ".webp", ".png")
        )
        if found:
            return found[0]

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download audio from YouTube (or any yt-dlp supported site) with high-quality "
                    "formats, metadata and cover art.",
        epilog="Presets:\n" + "\n".join(
            f"  {p.key:9s} {p.description}" for p in PRESETS.values()
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("urls", nargs="*", help="One or more video/playlist URLs or video IDs")
    parser.add_argument(
        "--preset",
        default="m4a",
        choices=list(PRESETS),
        help="Target audio format (default: m4a = bit-exact copy of YouTube's AAC stream)",
    )
    parser.add_argument(
        "--audio-lang",
        metavar="LANG",
        help="Select an audio language track by ISO code/prefix (e.g. es, en, es-MX); "
             "use yt-dlp -F URL to inspect available tracks",
    )
    parser.add_argument(
        "--trim-silence",
        action="store_true",
        help="Cut silence (5 s or longer) off the END of each file, without re-encoding",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Base output directory; files go in <dir>/<preset>/ (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch-file",
        help="Read URLs from a text file, one per line (blank lines and # comments ignored)",
    )
    parser.add_argument(
        "--playlist",
        action="store_true",
        help="Download every item when a URL points at a playlist (default: only the single video)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not write title/artist/album tags",
    )
    parser.add_argument(
        "--no-thumbnail",
        action="store_true",
        help="Do not embed the video thumbnail as cover art",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Re-download even if the video was already fetched (disables .downloaded.txt)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        type=browser_name,
        help="Use cookies from an installed browser for age-restricted/bot-checked videos "
             f"({', '.join(sorted(SUPPORTED_BROWSERS))}); case-insensitive",
    )
    parser.add_argument(
        "--cookies",
        metavar="FILE",
        help="Netscape-format cookies.txt (e.g. exported from your browser) — the fallback "
             "when --cookies-from-browser cannot read the browser's locked cookie file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without downloading anything",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal console output")

    args = parser.parse_args()

    urls = list(args.urls)
    if args.batch_file:
        batch_path = Path(args.batch_file)
        if not batch_path.exists():
            print(f"Error: batch file not found: {batch_path}")
            sys.exit(1)
        for line in batch_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    if not urls:
        parser.error("provide at least one URL or use --batch-file")
    if args.cookies and not Path(args.cookies).is_file():
        parser.error(f"cookies file not found: {args.cookies}")

    if args.dry_run:
        sys.exit(dry_run(urls, PRESETS[args.preset], args.playlist, args.quiet))

    results = download_audio(
        urls=urls,
        output_dir=args.output_dir,
        preset_key=args.preset,
        embed_metadata=not args.no_metadata,
        embed_thumbnail=not args.no_thumbnail,
        use_archive=not args.no_archive,
        cookies_from_browser=args.cookies_from_browser,
        cookies_file=args.cookies,
        playlist=args.playlist,
        quiet=args.quiet,
        audio_language=args.audio_lang,
        trim_silence=args.trim_silence,
    )
    # Already-downloaded inputs are a success, not an error.
    sys.exit(0 if results.ok else 1)


if __name__ == "__main__":
    main()
