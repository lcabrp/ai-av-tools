import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL

# Windows consoles default to cp1252, which crashes on emoji/CJK in video
# titles (same fix as benchmark_transcription.py).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

YOUTUBE_URL_PATTERNS = [
    r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})",
    r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/|live\/)([a-zA-Z0-9_-]{11})",
]


def extract_video_id(url_or_id: str) -> str | None:
    """Extract 11-character video ID from URL or raw ID string."""
    cleaned = url_or_id.strip()

    # Raw 11-character ID check
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", cleaned):
        return cleaned

    for pattern in YOUTUBE_URL_PATTERNS:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)

    return None


def is_youtube_input(value: str) -> bool:
    """Check if input is a YouTube URL or video ID."""
    return extract_video_id(value) is not None


def sanitize_filename(name: str) -> str:
    """Sanitize string to be safe for filenames across operating systems."""
    # Replace invalid chars with underscore
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", name)
    # Collapse multiple spaces or underscores
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized[:120]  # Limit length for OS safety


def fetch_video_title(video_id: str) -> str | None:
    """Attempt to fetch video title via YouTube oEmbed without requiring an API key."""
    oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    req = urllib.request.Request(
        oembed_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("title")
    except Exception:
        return None


def pick_caption_language(ytt: YouTubeTranscriptApi, video_id: str) -> str | None:
    """Choose the best available caption track before fetching.

    Without this, `ytt.fetch(video_id)` only asks for English and raises when the
    video has captions in another language, which used to push the whole pipeline
    into the much slower audio-download + faster-whisper fallback even though
    usable captions existed.

    Preference order: manual English, generated English, any manual track,
    any generated track (usually the original-language ASR).
    """
    try:
        lister = ytt.list if hasattr(ytt, "list") else ytt.list_transcripts
        available = list(lister(video_id))
    except Exception:
        return None

    if not available:
        return None

    def rank(transcript) -> tuple[int, int]:
        is_en = getattr(transcript, "language_code", "") == "en"
        is_generated = bool(getattr(transcript, "is_generated", False))
        return (0 if is_en else 1, 0 if not is_generated else 1)

    return getattr(min(available, key=rank), "language_code", None)


def fetch_youtube_transcript(
    url_or_id: str,
    output_dir: Path | str = "Output/transcripts",
    language: str | None = None,
    save_timestamps: bool = True,
) -> list[Path]:
    """Fetch transcripts from YouTube and save them to output_dir.

    Produces:
      - <stem>.txt: Clean continuous text for LLM summarization.
      - <stem>_timestamps.txt: Timestamped entries [MM:SS] for human reference.
      - <stem>.lang: Language code for downstream metadata.

    Returns:
      List containing the main transcript Path (.txt) for pipeline consumption.
    """
    video_id = extract_video_id(url_or_id)
    if not video_id:
        print(f"Error: Invalid YouTube URL or video ID: {url_or_id}")
        return []

    print(f"\nFetching YouTube transcript for video ID: {video_id}")

    # Fetch title for clean naming
    title = fetch_video_title(video_id)
    if title:
        file_stem = f"{sanitize_filename(title)} [{video_id}]"
    else:
        file_stem = f"youtube_{video_id}"

    # Initialize transcript API
    ytt = YouTubeTranscriptApi()
    transcript_snippets = None
    caption_language = language or pick_caption_language(ytt, video_id)
    detected_lang = caption_language or "en"

    try:
        if caption_language:
            transcript_snippets = ytt.fetch(video_id, languages=[caption_language, "en"])
        else:
            transcript_snippets = ytt.fetch(video_id)
    except Exception as e:
        # Fallback to get_transcript if older class API
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            try:
                transcript_snippets = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=[caption_language] if caption_language else ["en"]
                )
            except Exception as inner_e:
                print(f"Error retrieving transcript: {inner_e}")
                return []
        else:
            print(f"Error retrieving transcript: {e}")
            return []

    if not transcript_snippets:
        print(f"No transcript found for video: {video_id}")
        return []

    # Record the language the captions were actually delivered in (the .lang file
    # drives the default summary language downstream).
    detected_lang = getattr(transcript_snippets, "language_code", None) or detected_lang
    print(f"Using captions in '{detected_lang}'")

    # Extract text and timestamps
    plain_segments = []
    timestamp_lines = []

    for item in transcript_snippets:
        text = (
            item["text"]
            if isinstance(item, dict)
            else getattr(item, "text", "")
        )
        text = text.replace("\n", " ").strip()
        if not text:
            continue

        start_sec = int(
            item["start"]
            if isinstance(item, dict)
            else getattr(item, "start", 0)
        )
        m, s = divmod(start_sec, 60)
        h, m = divmod(m, 60)
        if h > 0:
            ts_str = f"[{h:02d}:{m:02d}:{s:02d}]"
        else:
            ts_str = f"[{m:02d}:{s:02d}]"

        plain_segments.append(text)
        timestamp_lines.append(f"{ts_str} {text}")

    full_plain_text = " ".join(plain_segments).strip()

    # Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Main plain text file (for LLM ingestion)
    output_file = out_path / f"{file_stem}.txt"
    output_file.write_text(full_plain_text, encoding="utf-8")
    print(f"Transcript saved to: {output_file}")

    # 2. Timestamped version
    if save_timestamps:
        ts_file = out_path / f"{file_stem}_timestamps.txt"
        ts_file.write_text("\n".join(timestamp_lines), encoding="utf-8")
        print(f"Timestamped transcript saved to: {ts_file}")

    # 3. Language metadata file
    lang_file = out_path / f"{file_stem}.lang"
    lang_file.write_text(detected_lang, encoding="utf-8")

    return [output_file]


def download_youtube_audio(
    url_or_id: str,
    output_dir: Path | str = "Docs/Audio/youtube",
) -> Path | None:
    """Download the best native audio stream of a YouTube video.

    Used as a fallback when a video has no subtitles to fetch. Downloads the
    native container (m4a/webm — no ffmpeg conversion needed; faster-whisper
    decodes them directly) and skips re-downloading if a file for the same
    video ID already exists.

    Returns:
      Path to the downloaded audio file, or None on failure.
    """
    video_id = extract_video_id(url_or_id)
    if not video_id:
        print(f"Error: Invalid YouTube URL or video ID: {url_or_id}")
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # NOTE: match "*<id>*" (not "*<id>.*") — yt-dlp names files
    # "<title> [<id>].<ext>", so the char after the ID is "]", not ".".
    existing = sorted(f for f in out_path.glob(f"*{video_id}*") if f.is_file())
    if existing:
        print(f"Reusing existing download: {existing[0]}")
        return existing[0]

    print(f"\nDownloading YouTube audio for video ID: {video_id}")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_path / "%(title)s [%(id)s].%(ext)s"),
        "noplaylist": True,
    }
    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as e:
        print(f"Error downloading YouTube audio: {e}")
        return None

    downloaded = sorted(f for f in out_path.glob(f"*{video_id}*") if f.is_file())
    if not downloaded:
        print(f"Download finished but no audio file found for video: {video_id}")
        return None
    print(f"Audio saved to: {downloaded[0]}")
    return downloaded[0]


def main():
    parser = argparse.ArgumentParser(
        description="Extract transcript from a YouTube video URL or ID."
    )
    parser.add_argument("url", help="YouTube video URL or 11-character video ID")
    parser.add_argument(
        "--output-dir",
        default="Output/transcripts",
        help="Directory to save generated transcripts (default: Output/transcripts)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Preferred language code for transcript (e.g. en, es; default: auto)",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Do not save the secondary timestamped file",
    )

    args = parser.parse_args()
    results = fetch_youtube_transcript(
        url_or_id=args.url,
        output_dir=args.output_dir,
        language=args.language,
        save_timestamps=not args.no_timestamps,
    )

    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()
