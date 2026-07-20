"""Staged orchestration: resume, modes, --jobs, GPU-stage serialization (§7, §12, §13, §16m8).

Stage callables (normalize/transcribe/diarize/merge/summarize/render_markdown) are
injected into Pipeline so the orchestration logic (resume, retry, mode dispatch,
GPU-stage serialization) can be unit-tested with fakes, independent of real
ffmpeg/mlx-whisper/pyannote/Ollama.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import naming
from .config import Config
from .logging_setup import per_file_log_path, setup_file_logger
from .manifest import Manifest
from .models import FileTask, ManifestEntry, RawDoc
from .stages import asr_mlx, audio as audio_stage, diarize as diarize_stage
from .stages import merge as merge_stage
from .stages import render as render_stage
from .stages import summarize as summarize_stage

_SENTINEL = object()


@dataclass
class RunOptions:
    mode: str  # "full" | "text" | "summary" | "resummarize" | "rerender"
    only: str | None = None
    skip: list[str] | None = None
    turbo: bool = False
    speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    names: list[str] | None = None
    frontmatter: bool = True
    wikilink_speakers: bool = False


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_hex(content_hash: str) -> str:
    return content_hash.split(":", 1)[-1]


def hash8(content_hash: str) -> str:
    return hash_hex(content_hash)[:8]


def filter_tasks(tasks: list[FileTask], only: str | None, skip: list[str] | None) -> list[FileTask]:
    result = [t for t in tasks if t.status != "skip"]
    if only:
        result = [t for t in result if t.path.stem == only or t.path.name == only]
    if skip:
        skip_set = set(skip)
        result = [t for t in result if t.path.name not in skip_set and t.path.stem not in skip_set]
    return result


def resolve_title_and_date(source_name: str, source_path: Path, doc: RawDoc) -> tuple[str, "date"]:
    stem = Path(source_name).stem
    day = naming.resolve_date(stem, source_path)
    if naming.is_technical_name(stem):
        title = doc.summary.title if (doc.summary and doc.summary.title) else naming.default_title_for_date(day)
    else:
        title = naming.normalize_title_from_name(stem)
    return title, day


def _json_default(obj):
    """Normalize values the stdlib json encoder can't handle.

    The ASR/diarization stages (mlx-whisper, pyannote, numpy 2.x) hand back numpy
    scalars — e.g. is_monologue ends up a numpy.bool, segment timestamps a
    numpy.float64 — which json refuses to serialize. Coerce them to native Python
    types. numpy is imported lazily so this module stays importable without it.
    """
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):  # numpy scalars and arrays both convert to native types
        return obj.tolist()
    item = getattr(obj, "item", None)
    if callable(item):  # other 0-d scalar-like values
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def load_raw_doc(raw_path: Path) -> RawDoc:
    with open(raw_path, "r", encoding="utf-8") as f:
        return RawDoc.from_dict(json.load(f))


def list_raw_files(systems_folder: Path) -> list[Path]:
    return sorted((Path(systems_folder) / "raw").glob("*.json"))


@dataclass
class _Ctx:
    task: FileTask
    log: logging.Logger
    log_path: Path
    started: float
    wav_path: Path | None = None
    duration: float = 0.0
    doc: RawDoc | None = None


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        manifest: Manifest,
        *,
        normalize=audio_stage.normalize,
        transcribe=asr_mlx.transcribe,
        diarize=diarize_stage.diarize,
        merge=merge_stage.merge,
        summarize=summarize_stage.summarize,
        render_markdown=render_stage.render_markdown,
    ):
        self.cfg = cfg
        self.manifest = manifest
        self.normalize = normalize
        self.transcribe = transcribe
        self.diarize = diarize
        self.merge = merge
        self.summarize = summarize
        self.render_markdown = render_markdown

    # --- shared helpers -------------------------------------------------

    def _mark_in_progress(self, content_hash: str, source_name: str, log_path: Path) -> None:
        """Transition to in_progress while preserving out_path from any prior run,
        so a later rerender/resummarize overwrites the same file instead of colliding."""
        existing = self.manifest.get(content_hash)
        self.manifest.upsert(
            ManifestEntry(
                content_hash=content_hash, source_name=source_name, status="in_progress",
                out_path=existing.out_path if existing else None,
                log_path=str(log_path), updated_at=utcnow_iso(),
            )
        )

    def _fail(self, task: FileTask, log_path: Path, log: logging.Logger, exc: Exception) -> None:
        log.info(f"FAILED: {exc}\n{traceback.format_exc()}")
        self.manifest.upsert(
            ManifestEntry(
                content_hash=task.content_hash,
                source_name=task.source_name,
                status="failed",
                log_path=str(log_path),
                error=str(exc),
                updated_at=utcnow_iso(),
            )
        )

    def _resolve_out_path(self, task: FileTask, day, title: str) -> Path:
        existing = self.manifest.get(task.content_hash)
        if existing and existing.out_path:
            return Path(existing.out_path)
        filename = naming.build_output_filename(day, title)
        return naming.resolve_collision(Path(self.cfg.out_folder), filename)

    def _write_outputs(self, doc: RawDoc, task: FileTask, opts: RunOptions, log: logging.Logger) -> tuple[Path, Path]:
        title, day = resolve_title_and_date(task.source_name, task.path, doc)

        raw_path = Path(self.cfg.systems_folder) / "raw" / f"{hash_hex(task.content_hash)}.json"
        atomic_write_json(raw_path, doc.to_dict())

        md = self.render_markdown(
            doc, day.isoformat(), title,
            frontmatter=opts.frontmatter,
            wikilink_speakers=opts.wikilink_speakers,
            long_form_from_min=self.cfg.long_form_from_min,
        )
        out_path = self._resolve_out_path(task, day, title)
        atomic_write_text(out_path, md)
        log.info(f"rendered: {out_path}")
        return out_path, raw_path

    # --- fresh audio (--text / full) staged pipeline --------------------

    def _safe_stage_a(self, task: FileTask, tmp_root: Path) -> _Ctx | None:
        h8 = hash8(task.content_hash)
        log_path = per_file_log_path(Path(self.cfg.logs_folder), task.source_name, h8)
        log = setup_file_logger(log_path)
        log.info(f"taken: {task.source_name} (hash {h8})")
        self._mark_in_progress(task.content_hash, task.source_name, log_path)
        ctx = _Ctx(task=task, log=log, log_path=log_path, started=time.monotonic())
        try:
            wav_path, duration = self.normalize(task.path, tmp_root)
            log.info(f"ffmpeg -> 16k mono wav, duration={duration:.1f}s")
            ctx.wav_path = wav_path
            ctx.duration = duration
            return ctx
        except Exception as exc:  # noqa: BLE001 - graceful per-file failure (§15)
            self._fail(task, log_path, log, exc)
            return None

    def _safe_stage_b(self, ctx: _Ctx, opts: RunOptions) -> _Ctx | None:
        task, log = ctx.task, ctx.log
        try:
            lang = self.cfg.asr_language.strip()
            language = None if lang.lower() in ("", "auto") else lang
            prompt = asr_mlx.build_initial_prompt(self.cfg.asr_prompt_extra)

            if opts.mode == "text":
                asr = self.transcribe(ctx.wav_path, opts.turbo, log, language=language, initial_prompt=prompt)
                doc = merge_stage.build_text_doc(
                    asr, content_hash=task.content_hash, source_name=task.source_name,
                    source_path=str(task.path), duration_sec=ctx.duration,
                )
            else:
                # Best-effort overlap: run diarization on a worker thread while ASR
                # runs here. Both read the same wav and are independent; merge waits
                # for both. Quality is unaffected (same models, same params).
                with ThreadPoolExecutor(max_workers=1) as diar_pool:
                    diar_future = diar_pool.submit(
                        self.diarize, ctx.wav_path, self.cfg.diarize_device,
                        opts.speakers, opts.min_speakers, opts.max_speakers, log,
                    )
                    asr = self.transcribe(ctx.wav_path, opts.turbo, log, language=language, initial_prompt=prompt)
                    diar = diar_future.result()
                doc = self.merge(
                    asr, diar, self.cfg.mono_threshold, opts.names, log,
                    content_hash=task.content_hash, source_name=task.source_name,
                    source_path=str(task.path), duration_sec=ctx.duration,
                    min_speaker_share=self.cfg.min_speaker_share,
                )
            ctx.doc = doc
            log.info(f"ASR done: language={doc.language}, segments={len(doc.segments)}")
            return ctx
        except Exception as exc:  # noqa: BLE001
            self._fail(task, ctx.log_path, log, exc)
            return None

    def _safe_stage_c(self, ctx: _Ctx, opts: RunOptions) -> None:
        task, log = ctx.task, ctx.log
        try:
            if opts.mode == "full":
                ctx.doc.summary = self.summarize(ctx.doc, self.cfg, log)
            out_path, raw_path = self._write_outputs(ctx.doc, task, opts, log)
            elapsed = time.monotonic() - ctx.started
            self.manifest.upsert(
                ManifestEntry(
                    content_hash=task.content_hash, source_name=task.source_name, status="done",
                    language=ctx.doc.language, num_speakers=ctx.doc.num_speakers,
                    duration_sec=ctx.doc.duration_sec, out_path=str(out_path), raw_path=str(raw_path),
                    log_path=str(ctx.log_path), elapsed_sec=elapsed, error=None, updated_at=utcnow_iso(),
                )
            )
            log.info(f"done (elapsed={elapsed:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            self._fail(task, ctx.log_path, log, exc)

    def run_all(self, tasks: list[FileTask], opts: RunOptions, jobs: int) -> None:
        """3-stage pipeline: [ffmpeg pool] -> [ASR+diarize, single GPU worker] -> [summary+render pool] (§13)."""
        if not tasks:
            return
        tmp_root = Path(tempfile.mkdtemp(prefix="transcriber-"))
        stage_a_out: queue.Queue = queue.Queue(maxsize=max(1, jobs))
        stage_b_out: queue.Queue = queue.Queue(maxsize=2)

        def worker_a() -> None:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                for fut in [pool.submit(self._safe_stage_a, t, tmp_root) for t in tasks]:
                    ctx = fut.result()
                    if ctx is not None:
                        stage_a_out.put(ctx)
            stage_a_out.put(_SENTINEL)

        def worker_b() -> None:
            while True:
                ctx = stage_a_out.get()
                if ctx is _SENTINEL:
                    break
                result = self._safe_stage_b(ctx, opts)
                if result is not None:
                    stage_b_out.put(result)
            stage_b_out.put(_SENTINEL)

        threads = [threading.Thread(target=worker_a), threading.Thread(target=worker_b)]
        for t in threads:
            t.start()

        pending = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            while True:
                ctx = stage_b_out.get()
                if ctx is _SENTINEL:
                    break
                pending.append(pool.submit(self._safe_stage_c, ctx, opts))
            for t in threads:
                t.join()
            for fut in pending:
                fut.result()

        shutil.rmtree(tmp_root, ignore_errors=True)

    # --- existing raw JSON (--summary / --resummarize / --rerender) -----

    def process_existing_raw(self, raw_path: Path, opts: RunOptions) -> None:
        doc = load_raw_doc(raw_path)
        h8 = hash8(doc.content_hash)
        log_path = per_file_log_path(Path(self.cfg.logs_folder), doc.source_name, h8)
        log = setup_file_logger(log_path)
        log.info(f"taken: {doc.source_name} (hash {h8}) mode={opts.mode}")
        started = time.monotonic()
        task = FileTask(
            path=Path(doc.source_path), content_hash=doc.content_hash,
            source_name=doc.source_name, status="redo", reason=opts.mode,
        )
        self._mark_in_progress(doc.content_hash, doc.source_name, log_path)
        try:
            if opts.mode != "rerender":
                doc.summary = self.summarize(doc, self.cfg, log)
                atomic_write_json(raw_path, doc.to_dict())
            out_path, raw_path_out = self._write_outputs(doc, task, opts, log)
            elapsed = time.monotonic() - started
            self.manifest.upsert(
                ManifestEntry(
                    content_hash=doc.content_hash, source_name=doc.source_name, status="done",
                    language=doc.language, num_speakers=doc.num_speakers, duration_sec=doc.duration_sec,
                    out_path=str(out_path), raw_path=str(raw_path_out), log_path=str(log_path),
                    elapsed_sec=elapsed, error=None, updated_at=utcnow_iso(),
                )
            )
            log.info(f"done (elapsed={elapsed:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            self._fail(task, log_path, log, exc)

    def run_existing(self, raw_paths: list[Path], opts: RunOptions, jobs: int) -> None:
        if not raw_paths:
            return
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(lambda p: self.process_existing_raw(p, opts), raw_paths))
