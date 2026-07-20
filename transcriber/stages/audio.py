"""ffmpeg normalization to 16k mono PCM WAV + ffprobe duration (§7)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FfmpegNotFoundError(RuntimeError):
    pass


def check_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise FfmpegNotFoundError("ffmpeg not found. Install with: brew install ffmpeg")


def normalize(src: Path, tmp_dir: Path) -> tuple[Path, float]:
    check_ffmpeg_available()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = tmp_dir / f"{src.stem}.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    duration = probe_duration(wav_path)
    return wav_path, duration


def probe_duration(wav_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(wav_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])
