"""Manifest: dedup by content hash, atomic writes, status tracking (§5.2/§12)."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from pathlib import Path

from .models import ManifestEntry

SCHEMA_VERSION = 1


class Manifest:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._entries: dict[str, ManifestEntry] = self._load()

    def _load(self) -> dict[str, ManifestEntry]:
        if not self._path.exists():
            return {}
        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            key: ManifestEntry(**value) for key, value in data.get("entries", {}).items()
        }

    def _save_locked(self) -> None:
        payload = {
            "schema": SCHEMA_VERSION,
            "entries": {key: asdict(entry) for key, entry in self._entries.items()},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._path)

    def get(self, content_hash: str) -> ManifestEntry | None:
        with self._lock:
            return self._entries.get(content_hash)

    def upsert(self, entry: ManifestEntry) -> None:
        """Insert or overwrite the entry for entry.content_hash (dedup key) and persist."""
        with self._lock:
            self._entries[entry.content_hash] = entry
            self._save_locked()

    def entries_with_status(self, status: str) -> list[ManifestEntry]:
        with self._lock:
            return [e for e in self._entries.values() if e.status == status]

    def all_entries(self) -> dict[str, ManifestEntry]:
        with self._lock:
            return dict(self._entries)
