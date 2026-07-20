"""mlx-whisper transcription wrapper (§7, §16 milestone 4).

mlx_whisper is imported lazily inside transcribe() so this module can be
imported (and unit-tested with a fake module) on machines without MLX/Metal.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..models import AsrResult, AsrSegment, AsrWord

FULL_REPO = "mlx-community/whisper-large-v3-mlx"
TURBO_REPO = "mlx-community/whisper-large-v3-turbo-mlx"


def transcribe(wav: Path, turbo: bool, log: logging.Logger) -> AsrResult:
    import mlx_whisper

    repo = TURBO_REPO if turbo else FULL_REPO
    log.info(f"ASR start: model={repo}")
    raw = mlx_whisper.transcribe(str(wav), path_or_hf_repo=repo, word_timestamps=True)
    segments = [_convert_segment(s) for s in raw["segments"]]
    language = raw.get("language", "unknown")
    log.info(f"ASR done: language={language}, segments={len(segments)}")
    return AsrResult(
        language=language,
        segments=segments,
        backend="mlx",
        model="large-v3-turbo" if turbo else "large-v3",
        turbo=turbo,
    )


def _convert_segment(raw_segment: dict) -> AsrSegment:
    words = [
        AsrWord(w=w["word"].strip(), start=w["start"], end=w["end"])
        for w in raw_segment.get("words", [])
    ]
    return AsrSegment(
        start=raw_segment["start"],
        end=raw_segment["end"],
        text=raw_segment["text"].strip(),
        words=words,
    )
