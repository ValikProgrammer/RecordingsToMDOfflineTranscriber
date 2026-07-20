"""Scan folder, hash content, diff against manifest -> to_do/skip/redo (§7/§12)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..manifest import Manifest
from ..models import FileTask

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".mp4", ".m4v", ".mov"}


def compute_blake2b(path: Path, chunk_size: int = 1 << 20) -> str:
    hasher = hashlib.blake2b()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return f"blake2b:{hasher.hexdigest()}"


def scan_audio_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def scan_and_hash(folder: Path, manifest: Manifest, retry_failed: bool = False) -> list[FileTask]:
    tasks: list[FileTask] = []
    for path in scan_audio_files(folder):
        content_hash = compute_blake2b(path)
        entry = manifest.get(content_hash)
        if entry is None:
            tasks.append(FileTask(path, content_hash, path.name, "to_do", "new file"))
        elif entry.status == "done":
            tasks.append(FileTask(path, content_hash, path.name, "skip", "already done"))
        elif entry.status == "in_progress":
            tasks.append(FileTask(path, content_hash, path.name, "redo", "unfinished previous run"))
        elif entry.status == "failed":
            if retry_failed:
                tasks.append(FileTask(path, content_hash, path.name, "redo", "retry failed"))
            else:
                tasks.append(
                    FileTask(path, content_hash, path.name, "skip", "failed previously; use --retry-failed")
                )
        else:  # pragma: no cover - defensive default for unknown status
            tasks.append(FileTask(path, content_hash, path.name, "to_do", "unknown manifest status"))
    return tasks
