"""Scan folder, hash content, diff against manifest -> to_do/skip/redo (§7/§12)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..manifest import Manifest
from ..models import FileTask, ManifestEntry, stage_status

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


def scan_and_hash(
    folder: Path,
    manifest: Manifest,
    *,
    retry_failed: bool = False,
    need_stage: str | None = None,
    force: bool = False,
) -> list[FileTask]:
    tasks: list[FileTask] = []
    for path in scan_audio_files(folder):
        content_hash = compute_blake2b(path)
        entry = manifest.get(content_hash)
        tasks.append(
            _classify(
                path, content_hash, entry, retry_failed=retry_failed, need_stage=need_stage, force=force
            )
        )
    return tasks


def _classify(
    path: Path,
    content_hash: str,
    entry: ManifestEntry | None,
    *,
    retry_failed: bool,
    need_stage: str | None,
    force: bool,
) -> FileTask:
    name = path.name
    if entry is None:
        return FileTask(path, content_hash, name, "to_do", "new file")

    if need_stage is None:
        return _classify_legacy(path, content_hash, entry, retry_failed)

    stage = stage_status(entry, need_stage)

    # Stage-level "already handled" wins over the coarse root status: a stage
    # can be done even while another stage on the same entry failed, and the
    # root `status=="done"` leftover from a legacy/other run must NOT skip a
    # stage that's actually still pending.
    if stage in ("done", "skipped") and not force:
        reason = f"{need_stage} already done" if stage == "done" else f"{need_stage} already skipped"
        return FileTask(path, content_hash, name, "skip", reason)

    if entry.status == "failed":
        if retry_failed:
            return FileTask(path, content_hash, name, "redo", "retry failed")
        return FileTask(path, content_hash, name, "skip", "failed previously; use --retry-failed")

    if entry.status == "in_progress" or stage == "in_progress":
        return FileTask(path, content_hash, name, "redo", "unfinished previous run")

    if stage == "failed":
        if retry_failed:
            return FileTask(path, content_hash, name, "redo", "retry failed")
        return FileTask(
            path, content_hash, name, "skip", f"{need_stage} failed previously; use --retry-failed"
        )

    if force and stage in ("done", "skipped"):
        return FileTask(path, content_hash, name, "redo", "force redo")

    return FileTask(path, content_hash, name, "to_do", f"{need_stage} pending")


def _classify_legacy(
    path: Path, content_hash: str, entry: ManifestEntry, retry_failed: bool
) -> FileTask:
    """Pre-stage-aware behavior, kept for callers that don't pass `need_stage`."""
    name = path.name
    if entry.status == "done":
        return FileTask(path, content_hash, name, "skip", "already done")
    if entry.status == "in_progress":
        return FileTask(path, content_hash, name, "redo", "unfinished previous run")
    if entry.status == "failed":
        if retry_failed:
            return FileTask(path, content_hash, name, "redo", "retry failed")
        return FileTask(path, content_hash, name, "skip", "failed previously; use --retry-failed")
    return FileTask(path, content_hash, name, "to_do", "unknown manifest status")  # pragma: no cover - defensive
