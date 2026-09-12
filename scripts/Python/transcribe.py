import argparse
import sys
import time
from pathlib import Path

import torch
from faster_whisper import WhisperModel

# Windows consoles default to cp1252, which crashes on emoji/CJK in file
# names (same fix as benchmark_transcription.py).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from scripts.Python.ops_log import log_operation
except ImportError:  # run directly as a script: siblings are top-level on sys.path
    from ops_log import log_operation

MEDIA_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv"
}


def is_media_file(path: Path) -> bool:

    return path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS


def resolve_device(requested: str) -> str:
    if requested in ("cuda", "cpu"):
        if requested == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available to torch; falling back to CPU.")
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_compute_type(device: str) -> str:
    """Pick the fastest compute type for the device based on benchmark findings.

    Ampere and newer (compute capability >= 8.0) have fast FP16 tensor cores;
    older architectures (e.g. T1000, compute capability 7.5) run ~3x faster
    with INT8. See Docs/whisper-benchmark.md.
    """
    if device != "cuda":
        return "int8"
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8:
        return "float16"
    return "int8_float16"


def transcribe_file(file_path: Path, model: WhisperModel, output_dir: Path,
                    language: str | None = None) -> Path:
    print(f"\nTranscribing: {file_path.name}")
    segments, info = model.transcribe(str(file_path), language=language, beam_size=5)
    transcript = " ".join(seg.text for seg in segments).strip()
    detected = info.language

    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = file_path.stem
    output_file = output_dir / f"{file_stem}.txt"
    output_file.write_text(transcript, encoding="utf-8")

    lang_file = output_dir / f"{file_stem}.lang"
    lang_file.write_text(detected, encoding="utf-8")

    print(f"Transcript saved to: {output_file}")
    print(f"Detected language: {detected} (probability {info.language_probability:.3f})")
    return output_file


def transcribe_media(input_path: str, model_size: str = "medium",
                     output_dir: str = "Output/transcripts",
                     device: str = "auto", compute_type: str = "auto",
                     language: str | None = None) -> list[Path]:
    target = Path(input_path)
    if not target.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)

    output_path = Path(output_dir)

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted([f for f in target.rglob("*") if is_media_file(f)])
        if not files:
            print(f"No supported media files found in: {input_path}")
            return []
        print(f"Found {len(files)} media file(s) in {input_path}")
    else:
        print(f"Error: Invalid path: {input_path}")
        sys.exit(1)

    resolved_device = resolve_device(device)
    resolved_compute = compute_type if compute_type != "auto" else default_compute_type(resolved_device)
    if resolved_device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu} (compute capability {capability[0]}.{capability[1]})")
    print(f"Loading faster-whisper model '{model_size}' (device={resolved_device}, "
          f"compute_type={resolved_compute})...")
    t_load_start = time.perf_counter()
    model = WhisperModel(model_size, device=resolved_device, compute_type=resolved_compute)
    load_seconds = time.perf_counter() - t_load_start
    config = f"faster-whisper {model_size} {resolved_compute} ({resolved_device})"

    generated = []
    for f in files:
        t0 = time.perf_counter()
        status = "ok"
        txt_path = None
        try:
            txt_path = transcribe_file(f, model, output_path, language)
        except Exception as e:
            status = f"{type(e).__name__}: {e}"
            print(f"Error transcribing {f.name}: {status}")
        elapsed = time.perf_counter() - t0
        chars = None
        if txt_path is not None:
            generated.append(txt_path)
            try:
                chars = len(txt_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        log_operation(
            operation="transcribe",
            input_file=f.name,
            config=config,
            elapsed_seconds=elapsed,
            output_file=txt_path.name if txt_path is not None else "",
            input_chars=None,
            output_chars=chars,
            load_seconds=load_seconds,
            status=status,
        )

    return generated


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video files using faster-whisper.")
    parser.add_argument("input", help="Path to audio/video file or directory")
    parser.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--output-dir",
        default="Output/transcripts",
        help="Directory to save transcripts (default: Output/transcripts)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device (default: auto = cuda when available)",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="CTranslate2 compute type (default: auto = float16 on compute "
             "capability >= 8.0, int8_float16 on older GPUs, int8 on CPU)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force language code (e.g. 'es'); default is auto-detect",
    )
    args = parser.parse_args()

    transcribe_media(args.input, args.model, args.output_dir, args.device,
                     args.compute_type, args.language)


if __name__ == "__main__":
    main()
