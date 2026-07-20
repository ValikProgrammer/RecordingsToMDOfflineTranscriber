"""Stub interface for future voice-ID (§14). v1: no-op; embeddings are already
captured into RawDoc.speakers_meta by stages/diarize.py so no rework is needed later."""
from __future__ import annotations

from pathlib import Path


class VoiceprintStore:
    def __init__(self, store_dir: Path):
        self._dir = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def enroll(self, name: str, embedding: list[float]) -> None:
        # TODO: voice-ID v2 — persist to systems/voiceprints/<name>.json
        pass

    def identify(self, embedding: list[float], threshold: float = 0.5) -> str | None:
        # TODO: voice-ID v2 — cosine-similarity search against enrolled embeddings
        return None
