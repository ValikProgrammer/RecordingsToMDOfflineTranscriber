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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import naming
from .config import Config
from .logging_setup import per_file_log_path, setup_file_logger
from .manifest import Manifest
from .models import FileTask, ManifestEntry, RawDoc
from .progress import NullReporter
from .stages import asr_mlx, audio as audio_stage, diarize as diarize_stage
from .stages import langdetect as langdetect_stage
from .stages import merge as merge_stage
from .stages import pretty as pretty_stage
from .stages import render as render_stage
from .stages import summarize as summarize_stage

_SENTINEL = object()


@dataclass
class RunOptions:
    mode: str  # "full" | "text" | "diarize" | "summary" | "resummarize" | "rerender"
    only: str | None = None
    skip: list[str] | None = None
    turbo: bool = False
    speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    names: list[str] | None = None
    frontmatter: bool = True
    wikilink_speakers: bool = False
    pretty: bool = False
    force: bool = False  # --summary --force: re-summarize raw docs that already have a summary


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


def _atomic_write(path: Path, write: "Callable[[object], None]") -> None:
    """Write via a per-call unique temp file, then os.replace onto `path`.

    The temp name MUST be unique per writer: a deterministic `path + ".tmp"`
    lets two threads writing the same target open the same tmp in "w" mode and
    interleave their bytes into one corrupt file (see issue #17). mkstemp gives
    each writer its own tmp in the destination dir (same filesystem, so replace
    stays atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: dict) -> None:
    _atomic_write(path, lambda f: json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default))


def atomic_write_text(path: Path, text: str) -> None:
    _atomic_write(path, lambda f: f.write(text))


def load_raw_doc(raw_path: Path) -> RawDoc:
    with open(raw_path, "r", encoding="utf-8") as f:
        return RawDoc.from_dict(json.load(f))


def list_raw_files(systems_folder: Path) -> list[Path]:
    return sorted((Path(systems_folder) / "raw").glob("*.json"))


def resolve_raw_by_query(systems_folder: Path, query: str) -> list[Path]:
    """Resolve a `--enroll-raw` argument to raw JSON paths.

    A direct path to an existing file wins. Otherwise treat `query` as a
    case-insensitive substring and match raw docs by filename stem (the content
    hash) or by their `source_name`."""
    direct = Path(query)
    if direct.is_file():
        return [direct]
    needle = query.lower()
    matches: list[Path] = []
    for p in list_raw_files(systems_folder):
        if needle in p.stem.lower():
            matches.append(p)
            continue
        try:
            if needle in load_raw_doc(p).source_name.lower():
                matches.append(p)
        except (OSError, ValueError, KeyError):  # skip unreadable/corrupt raw
            continue
    return matches


def filter_unsummarized(raw_paths: list[Path]) -> list[Path]:
    """Keep only raw docs that don't have a summary yet (incremental --summary).

    Mirrors scan_and_hash's skip-done: a raw doc gets a summary written back once
    summarized (full/--summary), so `summary is None` means "not summarized yet"."""
    return [p for p in raw_paths if load_raw_doc(p).summary is None]


@dataclass
class _Ctx:
    task: FileTask
    log: logging.Logger
    log_path: Path
    started: float
    idx: int = 0
    total: int = 0
    wav_path: Path | None = None
    duration: float = 0.0
    language: str | None = None  # resolved in stage A: forced code, or detected, or None (auto)
    doc: RawDoc | None = None


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        manifest: Manifest,
        *,
        normalize=audio_stage.normalize,
        transcribe=asr_mlx.transcribe,
        detect_language=langdetect_stage.detect_language,
        diarize=diarize_stage.diarize,
        merge=merge_stage.merge,
        summarize=summarize_stage.summarize,
        render_markdown=render_stage.render_markdown,
        pretty_transcript=pretty_stage.render_pretty_transcript,
        reporter=None,
        console_logs: bool = False,
    ):
        self.cfg = cfg
        self.manifest = manifest
        self.normalize = normalize
        self.transcribe = transcribe
        self.detect_language = detect_language
        self.diarize = diarize
        self.merge = merge
        self.summarize = summarize
        self.render_markdown = render_markdown
        self.pretty_transcript = pretty_transcript
        self.reporter = reporter if reporter is not None else NullReporter()
        self.console_logs = console_logs
        self._drain = threading.Event()  # set by request_drain(): finish in-flight, take no new files

    def request_drain(self) -> bool:
        """Ask run_all to stop taking NEW files while finishing in-flight ones.

        Returns True on the first request, False if a drain is already in progress
        (so a signal handler can force-quit on the second Ctrl-C)."""
        if self._drain.is_set():
            return False
        self._drain.set()
        return True

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
        self.reporter.file_failed(task.content_hash, exc)
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

        if opts.pretty:
            pretty_path = Path(self.cfg.out_folder) / "pretty" / out_path.name
            if pretty_path.exists() and not opts.force:
                # Incremental: the pretty rewrite is the expensive LLM step; skip it
                # when the output already exists. --force regenerates. Mirrors --summary.
                log.info(f"pretty exists, skipping (use --force to redo): {pretty_path}")
            else:
                pretty_body = self.pretty_transcript(doc, self.cfg, log)
                pretty_md = self.render_markdown(
                    doc, day.isoformat(), title,
                    frontmatter=opts.frontmatter,
                    wikilink_speakers=opts.wikilink_speakers,
                    long_form_from_min=self.cfg.long_form_from_min,
                    transcript_override=pretty_body,
                )
                atomic_write_text(pretty_path, pretty_md)
                log.info(f"pretty: {pretty_path}")

        return out_path, raw_path

    # --- fresh audio (--text / full) staged pipeline --------------------

    def _safe_stage_a(self, task: FileTask, idx: int, total: int, tmp_root: Path) -> _Ctx | None:
        h8 = hash8(task.content_hash)
        log_path = per_file_log_path(Path(self.cfg.logs_folder), task.source_name, h8)
        log = setup_file_logger(log_path, console=self.console_logs)
        log.info(f"taken: {task.source_name} (hash {h8})")
        self._mark_in_progress(task.content_hash, task.source_name, log_path)
        self.reporter.file_start(task.content_hash, idx, total, task.source_name, task.audio_sec)
        ctx = _Ctx(task=task, log=log, log_path=log_path, started=time.monotonic(), idx=idx, total=total)
        try:
            self.reporter.stage(task.content_hash, "FFMPEG", "normalizing")
            wav_path, duration = self.normalize(task.path, tmp_root)
            log.info(f"ffmpeg -> 16k mono wav, duration={duration:.1f}s")
            ctx.wav_path = wav_path
            ctx.duration = duration
            # Resolve the decode language here in stage A (CPU): when auto, detection
            # runs on this file while the GPU (stage B) transcribes an earlier one.
            lang = self.cfg.asr_language.strip()
            if lang.lower() in ("", "auto"):
                ctx.language = self.detect_language(wav_path, log, min_prob=self.cfg.lang_detect_min_prob)
            else:
                ctx.language = lang
            return ctx
        except Exception as exc:  # noqa: BLE001 - graceful per-file failure (§15)
            self._fail(task, log_path, log, exc)
            return None

    def _apply_voiceprints(self, doc: RawDoc, log: logging.Logger) -> None:
        """Enroll confirmed (--names) speakers, then auto-name the rest from the store."""
        from . import voiceprints

        store = voiceprints.VoiceprintStore(Path(self.cfg.systems_folder) / "voiceprints")
        voiceprints.enroll_named_speakers(doc, store)  # confirmed --names → ground truth
        voiceprints.identify_speakers(doc, store, self.cfg.voiceprint_threshold)

    def _safe_stage_b(self, ctx: _Ctx, opts: RunOptions) -> _Ctx | None:
        task, log = ctx.task, ctx.log
        try:
            language = ctx.language  # resolved in stage A (forced code, detected, or None=auto)
            prompt = asr_mlx.build_initial_prompt(self.cfg.asr_prompt_extra)

            def _run_asr():
                self.reporter.stage(task.content_hash, "WHISPER", "transcribing")
                result = self.transcribe(ctx.wav_path, opts.turbo, log, language=language, initial_prompt=prompt)
                result.segments = asr_mlx.filter_artifact_segments(
                    result.segments, self.cfg.asr_artifact_denylist_extra
                )
                return result

            if opts.mode == "text":
                asr = _run_asr()
                doc = merge_stage.build_text_doc(
                    asr, content_hash=task.content_hash, source_name=task.source_name,
                    source_path=str(task.path), duration_sec=ctx.duration,
                )
            else:
                self.reporter.stage(task.content_hash, "DIARIZE", "diarizing")
                # Best-effort overlap: run diarization on a worker thread while ASR
                # runs here. Both read the same wav and are independent; merge waits
                # for both. Quality is unaffected (same models, same params).
                with ThreadPoolExecutor(max_workers=1) as diar_pool:
                    diar_future = diar_pool.submit(
                        self.diarize, ctx.wav_path, self.cfg.diarize_device,
                        opts.speakers, opts.min_speakers, opts.max_speakers, log,
                    )
                    asr = _run_asr()
                    diar = diar_future.result()
                self.reporter.stage(task.content_hash, "MERGE", "merging speakers")
                doc = self.merge(
                    asr, diar, self.cfg.mono_threshold, opts.names, log,
                    content_hash=task.content_hash, source_name=task.source_name,
                    source_path=str(task.path), duration_sec=ctx.duration,
                    min_speaker_share=self.cfg.min_speaker_share,
                )
                if self.cfg.voiceprint_enabled:
                    self.reporter.stage(task.content_hash, "VOICEID", "matching voices")
                    self._apply_voiceprints(doc, log)
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
                self.reporter.stage(task.content_hash, "SUMMARY", "summarizing")
                ctx.doc.summary = self.summarize(ctx.doc, self.cfg, log)
            self.reporter.stage(task.content_hash, "RENDER", "rendering")
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
            self.reporter.file_done(task.content_hash, elapsed, out_path)
        except Exception as exc:  # noqa: BLE001
            self._fail(task, ctx.log_path, log, exc)

    def run_all(self, tasks: list[FileTask], opts: RunOptions, jobs: int) -> None:
        """3-stage pipeline: [ffmpeg pool] -> [ASR+diarize, single GPU worker] -> [summary+render pool] (§13)."""
        if not tasks:
            return
        tmp_root = Path(tempfile.mkdtemp(prefix="transcriber-"))
        stage_a_out: queue.Queue = queue.Queue(maxsize=max(1, jobs))
        stage_b_out: queue.Queue = queue.Queue(maxsize=2)

        total = len(tasks)

        def worker_a() -> None:
            # Bound how far stage A (ffmpeg + lang-detect, CPU) may run ahead of the
            # GPU stage: submit at most `stage_a_lookahead` files at a time instead of
            # all at once. Otherwise the pool churns through the whole batch eagerly,
            # piling up normalized files + model work and driving the machine into
            # memory pressure / OOM (issue: night-run jetsam kills).
            lookahead = max(1, self.cfg.stage_a_lookahead)
            task_iter = iter(enumerate(tasks, start=1))
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                pending: set = set()
                exhausted = False
                while pending or not exhausted:
                    while len(pending) < lookahead and not exhausted:
                        if self._drain.is_set():  # graceful stop: no new files, let in-flight finish
                            exhausted = True
                            break
                        nxt = next(task_iter, None)
                        if nxt is None:
                            exhausted = True
                            break
                        i, t = nxt
                        pending.add(pool.submit(self._safe_stage_a, t, i, total, tmp_root))
                    if not pending:
                        break
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
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

    def process_existing_raw(self, raw_path: Path, opts: RunOptions, idx: int = 1, total: int = 1) -> None:
        doc = load_raw_doc(raw_path)
        h8 = hash8(doc.content_hash)
        log_path = per_file_log_path(Path(self.cfg.logs_folder), doc.source_name, h8)
        log = setup_file_logger(log_path, console=self.console_logs)
        log.info(f"taken: {doc.source_name} (hash {h8}) mode={opts.mode}")
        started = time.monotonic()
        task = FileTask(
            path=Path(doc.source_path), content_hash=doc.content_hash,
            source_name=doc.source_name, status="redo", reason=opts.mode,
            audio_sec=doc.duration_sec,
        )
        self._mark_in_progress(doc.content_hash, doc.source_name, log_path)
        self.reporter.file_start(doc.content_hash, idx, total, doc.source_name, doc.duration_sec)
        try:
            if opts.mode != "rerender":
                self.reporter.stage(doc.content_hash, "SUMMARY", "summarizing")
                doc.summary = self.summarize(doc, self.cfg, log)
                atomic_write_json(raw_path, doc.to_dict())
            self.reporter.stage(doc.content_hash, "RENDER", "rendering")
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
            self.reporter.file_done(doc.content_hash, elapsed, out_path)
        except Exception as exc:  # noqa: BLE001
            self._fail(task, log_path, log, exc)

    def run_existing(self, raw_paths: list[Path], opts: RunOptions, jobs: int) -> None:
        if not raw_paths:
            return
        total = len(raw_paths)
        indexed = list(enumerate(raw_paths, start=1))
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(lambda it: self.process_existing_raw(it[1], opts, it[0], total), indexed))
