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

DEFAULT_INITIAL_PROMPT = (
    "Совещание. Обсуждаем ФизТех, хакатон, стипендию, ментора, практику, "
    "дедлайн, проект, репозиторий, коммит, деплой, бэкенд, фронтенд, API, "
    "Телеграм-бота, субботнюю школу."
)


def build_initial_prompt(extra: str) -> str:
    """Return the built-in glossary prompt, with the user's extra terms appended."""
    extra = (extra or "").strip()
    return f"{DEFAULT_INITIAL_PROMPT} {extra}".strip() if extra else DEFAULT_INITIAL_PROMPT


def transcribe(
    wav: Path,
    turbo: bool,
    log: logging.Logger,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> AsrResult:
    import mlx_whisper

    repo = TURBO_REPO if turbo else FULL_REPO
    log.info(f"ASR start: model={repo}")
    kwargs = dict(
        path_or_hf_repo=repo,
        word_timestamps=True,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
        initial_prompt=initial_prompt,
    )
    if language:
        kwargs["language"] = language
    raw = mlx_whisper.transcribe(str(wav), **kwargs)
    segments = [_convert_segment(s) for s in raw["segments"]]
    language_detected = raw.get("language", "unknown")
    log.info(f"ASR done: language={language_detected}, segments={len(segments)}")
    return AsrResult(
        language=language_detected,
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
