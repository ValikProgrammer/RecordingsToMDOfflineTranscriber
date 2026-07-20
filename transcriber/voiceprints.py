"""Voice-ID (§14): enroll speaker embeddings and identify speakers across recordings.

Embeddings are captured into RawDoc.speakers_meta by stages/diarize.py. This module
persists them per name under systems/voiceprints/<slug>.json and matches new speakers
by cosine similarity against the mean enrolled embedding.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _slug(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name.strip().lower(), flags=re.UNICODE).strip("_") or "unnamed"


class VoiceprintStore:
    def __init__(self, store_dir: Path):
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self._dir / f"{_slug(name)}.json"

    def enroll(self, name: str, embedding: list[float]) -> None:
        if not embedding:
            return
        path = self._path(name)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {"name": name, "embeddings": []}
        data["embeddings"].append([float(x) for x in embedding])
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _enrolled_means(self):
        for p in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            embs = data.get("embeddings") or []
            if not embs:
                continue
            dim = len(embs[0])
            mean = [sum(e[i] for e in embs) / len(embs) for i in range(dim)]
            yield data.get("name", p.stem), mean

    def identify(self, embedding: list[float], threshold: float = 0.7) -> str | None:
        if not embedding:
            return None
        best_name, best_sim = None, 0.0
        for name, mean in self._enrolled_means():
            sim = cosine_similarity(embedding, mean)
            if sim > best_sim:
                best_name, best_sim = name, sim
        return best_name if best_sim >= threshold else None


def identify_speakers(doc, store: VoiceprintStore, threshold: float = 0.7):
    """Fill in names for still-unnamed speakers by matching their embedding against
    the voiceprint store, then relabel segments + speakers_meta accordingly."""
    from .stages.merge import _apply_name_map

    name_map: dict[str, str] = {}
    for sm in doc.speakers_meta:
        if sm.name is None and sm.embedding:
            match = store.identify(sm.embedding, threshold)
            if match:
                name_map[sm.label] = match
    if not name_map:
        return doc
    doc.segments = _apply_name_map(doc.segments, name_map)
    for sm in doc.speakers_meta:
        if sm.label in name_map:
            sm.name = name_map[sm.label]
    return doc
