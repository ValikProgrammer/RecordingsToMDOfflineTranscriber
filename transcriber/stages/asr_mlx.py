"""mlx-whisper transcription wrapper (§7, §16 milestone 4).

mlx_whisper is imported lazily inside transcribe() so this module can be
imported (and unit-tested with a fake module) on machines without MLX/Metal.
"""
from __future__ import annotations

import logging
import re
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


# Known Whisper hallucinations learned from YouTube-caption training data. These
# surface as standalone segments on low-speech audio. Phrases here are dropped
# only when a segment consists ENTIRELY of them (see filter_artifact_segments).
DEFAULT_ARTIFACT_DENYLIST_EXACT = (
    "продолжение следует",
    "спасибо за просмотр",
    "спасибо за внимание",
    "подписывайтесь на канал",
    "ставьте лайки",
    "продолжение в следующем видео",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "takk for watching",
)
# Credit-line prefixes; a name/initials usually follow, so match on the prefix.
DEFAULT_ARTIFACT_DENYLIST_PREFIX = (
    "субтитры сделал",
    "субтитры делал",
    "субтитры создавал",
    "субтитры подготовил",
    "субтитры и перевод",
    "редактор субтитров",
    "корректор",
)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_artifact(text: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT_RE.sub(" ", text.lower())).strip()


def filter_artifact_segments(segments: list[AsrSegment], extra=()) -> list[AsrSegment]:
    """Drop segments that are purely a known Whisper hallucination.

    A segment is dropped if, after removing every exact denylist phrase, nothing
    is left (handles repeats like "thank you for watching thank you for watching"),
    or if its text starts with a credit-line prefix. Real speech is never touched.
    """
    exact = [n for n in (_normalize_artifact(p) for p in (*DEFAULT_ARTIFACT_DENYLIST_EXACT, *extra)) if n]
    prefixes = [n for n in (_normalize_artifact(p) for p in DEFAULT_ARTIFACT_DENYLIST_PREFIX) if n]
    kept: list[AsrSegment] = []
    for seg in segments:
        norm = _normalize_artifact(seg.text)
        if not norm:
            kept.append(seg)
            continue
        if any(norm.startswith(p) for p in prefixes):
            continue
        remainder = norm
        for phrase in exact:
            remainder = remainder.replace(phrase, " ")
        if not remainder.strip():
            continue
        kept.append(seg)
    return kept


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
