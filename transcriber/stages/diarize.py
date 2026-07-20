"""pyannote diarization wrapper: MPS with CPU fallback, per-speaker embeddings (§7, §9, §14).

pyannote.audio/torch are imported lazily inside diarize() so this module can be
imported (and its pure mapping logic unit-tested) without those heavy deps installed.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from ..models import DiarResult, DiarSegment

PIPELINE_NAME = "pyannote/speaker-diarization-3.1"


class MissingHfTokenError(RuntimeError):
    pass


def diarize(
    wav: Path,
    device: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    log: logging.Logger,
) -> DiarResult:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise MissingHfTokenError(
            "HF_TOKEN is not set. Get a token at huggingface.co, accept the "
            f"{PIPELINE_NAME} model terms on its page, and save the token as HF_TOKEN in .env."
        )

    from pyannote.audio import Pipeline
    import torch

    pipeline = Pipeline.from_pretrained(PIPELINE_NAME, token=token)
    resolved_device = device
    if device == "mps" and not torch.backends.mps.is_available():
        log.info("MPS unavailable, falling back to CPU")
        resolved_device = "cpu"
    pipeline.to(torch.device(resolved_device))

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    output = pipeline(str(wav), **kwargs)
    return build_diar_result(output, log)


def build_diar_result(output, log: logging.Logger) -> DiarResult:
    # pyannote.audio 4.x returns a DiarizeOutput dataclass; older versions (and the
    # pipeline's legacy mode) return the Annotation directly. Unwrap either way.
    annotation = getattr(output, "speaker_diarization", output)

    segments: list[DiarSegment] = []
    total_speech: dict[str, float] = {}
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(DiarSegment(start=turn.start, end=turn.end, speaker=speaker))
        total_speech[speaker] = total_speech.get(speaker, 0.0) + (turn.end - turn.start)

    embeddings = extract_embeddings(output, annotation, log)
    log.info(f"diarize done: raw_speakers={len(total_speech)}")
    return DiarResult(segments=segments, embeddings=embeddings, total_speech_sec=total_speech)


def extract_embeddings(output, annotation, log: logging.Logger) -> dict[str, list[float]]:
    # pyannote.audio 4.x computes one embedding per speaker during diarization and
    # returns them on the output as a (num_speakers, dimension) array, ordered to
    # match annotation.labels().
    vectors = getattr(output, "speaker_embeddings", None)
    if vectors is None:
        log.info("no speaker embeddings available in the pipeline output, skipping embeddings")
        return {}

    embeddings: dict[str, list[float]] = {}
    for index, speaker in enumerate(annotation.labels()):
        try:
            embeddings[speaker] = [float(value) for value in vectors[index]]
        except Exception as exc:  # noqa: BLE001 - one bad speaker shouldn't fail the whole doc
            log.info(f"failed to extract embedding for {speaker}: {exc}")
    return embeddings
