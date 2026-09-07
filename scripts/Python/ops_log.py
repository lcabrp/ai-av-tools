"""Always-on operation log: append one row per transcribe/summarize run.

Unlike the benchmark scripts (which time controlled slices for GPU comparison),
this logs every real production operation so GPU speed/appropriateness can be
judged from history. Logging never raises: failures are printed, not thrown.
Shared CSV lives at Docs/operations-log.csv (tracked, portable across PCs).
"""

import csv
import socket
import subprocess
from datetime import datetime
from pathlib import Path

OPS_LOG_FIELDS = [
    "recorded_at", "machine", "gpu", "vram_gb",
    "operation", "input_file", "config",
    "load_seconds", "elapsed_seconds",
    "input_chars", "output_chars", "status", "output_file",
]

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "Docs" / "operations-log.csv"


def _gpu_info() -> dict:
    info = {"gpu": "none", "vram_gb": ""}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["vram_gb"] = f"{props.total_memory / 1024**3:.1f}"
            return info
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        parts = out.stdout.strip().splitlines()[0].split(",")
        info["gpu"] = parts[0].strip()
        info["vram_gb"] = f"{float(parts[1].strip()) / 1024:.1f}"
    except Exception:
        pass
    return info


def _fmt(value) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else ""


def log_operation(
    operation: str,
    input_file: str,
    config: str,
    elapsed_seconds: float | None,
    output_file: str = "",
    input_chars: int | None = None,
    output_chars: int | None = None,
    load_seconds: float | None = None,
    status: str = "ok",
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    """Append one row to the operations log. Never raises."""
    try:
        gpu = _gpu_info()
        row = {
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "machine": socket.gethostname(),
            "gpu": gpu["gpu"],
            "vram_gb": gpu["vram_gb"],
            "operation": operation,
            "input_file": input_file,
            "config": config,
            "load_seconds": _fmt(load_seconds),
            "elapsed_seconds": _fmt(elapsed_seconds),
            "input_chars": str(input_chars) if input_chars is not None else "",
            "output_chars": str(output_chars) if output_chars is not None else "",
            "status": status,
            "output_file": output_file,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = True
        if log_path.exists() and log_path.stat().st_size > 0:
            with log_path.open(newline="", encoding="utf-8") as f:
                needs_header = csv.DictReader(f).fieldnames != OPS_LOG_FIELDS
            if needs_header:
                # Unexpected columns: preserve existing rows after the header fix.
                with log_path.open(newline="", encoding="utf-8") as f:
                    existing = list(csv.DictReader(f))
                with log_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=OPS_LOG_FIELDS)
                    writer.writeheader()
                    writer.writerows(existing)
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OPS_LOG_FIELDS)
            if needs_header and f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"Warning: could not write operations log: {e}")
