"""faster-whisper (CTranslate2) ASR backend — CPU path with beam search.

Same interface as asr_mlx.transcribe so it drops straight into Pipeline. Runs on
CPU (CTranslate2 has no Metal backend), which is slower than the mlx/Metal path
but supports beam search — the point of the A/B comparison. faster_whisper is
imported lazily so this module stays importable without the dependency.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..models import AsrResult, AsrSegment, AsrWord

FULL_MODEL = "large-v3"
TURBO_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"


def transcribe(
    wav: Path,
    turbo: bool,
    log: logging.Logger,
    language: str | None = None,
    initial_prompt: str | None = None,
    beam_size: int = 5,
    compute_type: str = "int8",
) -> AsrResult:
    from faster_whisper import WhisperModel

    model_name = TURBO_MODEL if turbo else FULL_MODEL
    log.info(f"ASR start: faster-whisper model={model_name} device=cpu beam={beam_size} compute={compute_type}")
    model = WhisperModel(model_name, device="cpu", compute_type=compute_type)

    segments_iter, info = model.transcribe(
        str(wav),
        language=language,
        beam_size=beam_size,
        initial_prompt=initial_prompt,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    segments = [_convert_segment(s) for s in segments_iter]
    language_detected = language or info.language
    log.info(f"ASR done: language={language_detected}, segments={len(segments)}")
    return AsrResult(
        language=language_detected,
        segments=segments,
        backend="faster-whisper",
        model="large-v3-turbo" if turbo else "large-v3",
        turbo=turbo,
    )


def _convert_segment(seg) -> AsrSegment:
    words = [
        AsrWord(w=w.word.strip(), start=w.start, end=w.end)
        for w in (seg.words or [])
    ]
    return AsrSegment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words)
