import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

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
    detected_lang = language or "en"

    try:
        if language:
            transcript_snippets = ytt.fetch(video_id, languages=[language, "en"])
        else:
            transcript_snippets = ytt.fetch(video_id)
    except Exception as e:
        # Fallback to get_transcript if older class API
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            try:
                transcript_snippets = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=[language] if language else ["en"]
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
