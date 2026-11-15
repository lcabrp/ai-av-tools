"""File a run's artifacts into its topic folder (e.g. exploring-pathways/Pathways/...).

av-tools writes transcripts to Output/transcripts and summaries to Output/. Those
are staging locations: every recording belongs to a topic folder in the downstream
knowledge repo, so after a run its artifacts are moved there:

    <stem>.txt, <stem>.lang, <stem>_timestamps.txt   (from Output/transcripts)
    <stem>_summary_*.md                              (from Output)

Moving is the default so Output/ stays clean (no duplicate copies of the same
summary). Use --keep-source to copy instead. Re-running is safe: files already in
the destination with the same size and mtime are left alone.

Usage:
    uv run scripts/Python/file_outputs.py "LIVE PREPARATORIO [ke_fvNlrVnE]" \
        --dest "C:\\...\\ky-tax-preparer\\course-archives\\HugoTaxPro\\Preparatorio"

Tax-preparer recordings go to the ky-tax-preparer repo; every other topic goes to
exploring-pathways\\Pathways\\<Topic>\\<Brand>\\<Clase|Date>.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Windows consoles default to cp1252, which crashes on emoji/accents in file names
# (same fix as summarize.py and benchmark_transcription.py).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent


def collect_sources(stem: str, transcripts_dir: Path, output_dir: Path) -> list[Path]:
    """Return the artifacts for one run, in a stable order.

    Plain string prefix matching instead of Path.glob: transcript names contain
    brackets (e.g. "[ke_fvNlrVnE]"), which glob treats as a character class.
    """
    candidates = [
        transcripts_dir / f"{stem}.txt",
        transcripts_dir / f"{stem}.lang",
        transcripts_dir / f"{stem}_timestamps.txt",
    ]
    prefix = f"{stem}_summary_"
    if output_dir.exists():
        summaries = [
            p for p in output_dir.iterdir()
            if p.is_file() and p.name.startswith(prefix) and p.suffix == ".md"
        ]
        candidates += sorted(summaries)
    return candidates


def same_file(src: Path, dst: Path) -> bool:
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        return False
    # copy2/move preserve mtime, so equal mtimes mean the destination is current.
    return int(dst.stat().st_mtime) == int(src.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move a run's transcripts and summaries into their topic folder."
    )
    parser.add_argument("stem", help='Transcript base name, e.g. "LIVE PREPARATORIO [ke_fvNlrVnE]"')
    parser.add_argument("--dest", required=True, help="Destination topic folder (created if missing)")
    parser.add_argument("--transcripts-dir", default=str(ROOT / "Output" / "transcripts"))
    parser.add_argument("--output-dir", default=str(ROOT / "Output"))
    parser.add_argument("--keep-source", action="store_true", help="Copy instead of moving (leaves Output/ in place)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    args = parser.parse_args()

    stem = args.stem.strip()
    dest = Path(args.dest).expanduser().resolve()
    sources = collect_sources(stem, Path(args.transcripts_dir), Path(args.output_dir))
    existing = [p for p in sources if p.exists()]

    verb = "copy" if args.keep_source else "move"
    if not existing:
        archived = (
            [p.name for p in dest.iterdir() if p.is_file() and p.name.startswith(stem)]
            if dest.exists() else []
        )
        if archived:
            print(f"Nothing to file: '{stem}' is already archived in {dest} ({len(archived)} file(s)).")
            return 0
        print(f"Nothing to file: no artifacts found for '{stem}'.")
        print(f"Searched: {args.transcripts_dir} and {args.output_dir}")
        return 1

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    for src in existing:
        target = dest / src.name
        if same_file(src, target):
            print(f"  up to date  {src.name}")
            skipped += 1
            if not args.keep_source and not args.dry_run:
                src.unlink()
            continue
        if args.dry_run:
            print(f"  would {verb}  {src.name}")
        elif args.keep_source:
            shutil.copy2(src, target)
            print(f"  copied      {src.name}")
        else:
            if target.exists():
                # Different content at the destination: replace it.
                target.unlink()
            shutil.move(str(src), str(target))
            print(f"  moved       {src.name}")
        done += 1

    expected = [
        Path(args.transcripts_dir) / f"{stem}.txt",
        Path(args.transcripts_dir) / f"{stem}.lang",
        Path(args.transcripts_dir) / f"{stem}_timestamps.txt",
    ]
    missing = [p.name for p in expected if not p.exists() and not (dest / p.name).exists()]

    print(f"\nDestination: {dest}")
    print(f"{verb.capitalize()}d {done}, unchanged {skipped}, missing {len(missing)}")
    if missing:
        print("Missing (never produced or already archived): " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
