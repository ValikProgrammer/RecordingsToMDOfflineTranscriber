"""Sample-based mono pre-check to skip full diarization on single-speaker files.

Full diarization (pyannote) is the slowest stage in the pipeline. Many recordings
are monologues (a single speaker dictating notes, a voice memo, etc.) where
diarization only adds overhead with no useful output. Instead of always running
the full pipeline, we sample a handful of windows across the file, diarize just
those windows, and decide:
  - every sampled window sees <=1 speaker -> likely monologue, caller may skip
    the full diarization pass and label everything as a single speaker
  - any window sees >=2 speakers         -> not a monologue, run full diarize
  - every window fails                    -> inconclusive, caller falls back to
                                             running full diarize (safe default)

Short files (<= SHORT_FULL_SEC) are cheap to diarize in full, so we sample the
whole file as a single window rather than a 30s slice, for a more reliable read.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from ..models import DiarResult
from .langdetect import window_starts

WINDOW_SEC = 30
SHORT_FULL_SEC = 300
NUM_WINDOWS = 3

DiarizeFn = Callable[[Path, str, int | None, int | None, int | None, logging.Logger], DiarResult]


def decide_mono(counts: list[int]) -> bool | None:
    """Pure vote over per-window speaker counts.

    Empty input (every window failed) is inconclusive -> None, so callers fall
    back to running the full diarization pass rather than guessing.
    """
    if not counts:
        return None
    return all(c <= 1 for c in counts)


def _window_plan(duration: float) -> list[tuple[float, float]]:
    """Return (start, length) pairs of windows to sample for the pre-check."""
    if duration <= SHORT_FULL_SEC:
        return [(0.0, duration)]
    starts = window_starts(duration, NUM_WINDOWS, WINDOW_SEC)
    return [(s, WINDOW_SEC) for s in starts]


def _extract_window(wav: Path, start: float, length: float, out: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", str(length), "-i", str(wav),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)],
        check=True, capture_output=True,
    )
    return out


def _speaker_count(diar: DiarResult) -> int:
    if diar.total_speech_sec:
        return len(diar.total_speech_sec)
    return len({seg.speaker for seg in diar.segments})


def is_likely_monologue(
    wav: Path, duration: float, diarize_fn: DiarizeFn, device: str, log: logging.Logger
) -> bool | None:
    """Sample a few windows and decide whether `wav` is likely a single-speaker file.

    Returns True (mono, safe to skip full diarize), False (multiple speakers seen,
    run full diarize), or None (every window failed -> caller should fall back to
    running full diarize).
    """
    windows = _window_plan(duration)
    counts: list[int] = []
    with tempfile.TemporaryDirectory(prefix="mono-precheck-") as td:
        for i, (start, length) in enumerate(windows):
            window_wav = Path(td) / f"win{i}.wav"
            try:
                _extract_window(wav, start, length, window_wav)
                diar = diarize_fn(window_wav, device, None, None, None, log)
                count = _speaker_count(diar)
                log.info(f"mono-precheck: window@{start:.1f}s -> {count} speaker(s)")
                counts.append(count)
            except Exception as exc:  # noqa: BLE001 - one bad window shouldn't kill the pre-check
                log.info(f"mono-precheck: window@{start:.1f}s failed ({exc})")

    decision = decide_mono(counts)
    log.info(f"mono-precheck: decision={decision}")
    return decision
