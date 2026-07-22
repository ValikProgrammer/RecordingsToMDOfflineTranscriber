"""Entry point: python -m transcriber (§6, §16 milestones 1 & 10)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from .cli import apply_overrides, build_run_options, parse_args, resolve_mode
from .config import Config, load_config
from .logging_setup import setup_run_logger
from .manifest import Manifest
from .pipeline import Pipeline, filter_tasks, list_raw_files
from .progress import make_reporter
from .stages.audio import FfmpegNotFoundError, check_ffmpeg_available, normalize, probe_duration
from .stages.ingest import scan_and_hash


def probe_total_audio(tasks, log) -> float:
    """ffprobe each source up front to size the progress bar. A probe failure
    leaves that file's duration at 0 (it just won't advance the bar)."""
    total = 0.0
    for task in tasks:
        try:
            task.audio_sec = probe_duration(Path(task.path))
        except Exception as exc:  # noqa: BLE001 - non-fatal, bar just won't count it
            log.info(f"duration probe failed for {task.source_name}: {exc}")
            task.audio_sec = 0.0
        total += task.audio_sec
    return total


def cmd_warmup(cfg: Config, log) -> int:
    try:
        check_ffmpeg_available()
    except FfmpegNotFoundError as exc:
        print(exc)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        src_dir = Path(tmp) / "src"
        work_dir = Path(tmp) / "work"
        src_dir.mkdir()
        silent_src = src_dir / "silence.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "2", str(silent_src)],
            check=True, capture_output=True,
        )
        wav_path, _ = normalize(silent_src, work_dir)

        try:
            from .stages.asr_mlx import transcribe
            transcribe(wav_path, turbo=False, log=log)
        except ModuleNotFoundError as exc:
            print(f"mlx-whisper is not installed ({exc}). Run setup_mac.sh.")
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"ASR warmup failed: {exc}")
            return 1

        try:
            from .stages.diarize import diarize
            diarize(wav_path, cfg.diarize_device, None, None, None, log)
        except ModuleNotFoundError as exc:
            print(f"pyannote.audio is not installed ({exc}). Run setup_mac.sh.")
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"diarize warmup failed: {exc}")
            return 1

    print("Models are ready.")
    return 0


def cmd_dry_run(cfg: Config, args) -> int:
    manifest = Manifest(Path(cfg.systems_folder) / "manifest.json")
    tasks = scan_and_hash(Path(cfg.input_folder), manifest, retry_failed=args.retry_failed)
    if not tasks:
        print(f"No supported audio files found in {cfg.input_folder}.")
        return 0
    for task in tasks:
        print(f"{task.status:5}  {task.source_name}  ({task.reason})")
    return 0


def _raw_duration(raw_path: Path) -> float:
    try:
        import json

        with open(raw_path, "r", encoding="utf-8") as f:
            return float(json.load(f).get("duration_sec", 0.0))
    except Exception:  # noqa: BLE001 - missing/corrupt raw just doesn't size the bar
        return 0.0


def _select_transcribe(cfg: Config, args):
    """Pick the ASR backend callable, matching asr_mlx.transcribe's interface."""
    if cfg.asr_backend == "faster-whisper":
        from functools import partial

        from .stages import asr_faster
        log_msg = f"ASR backend: faster-whisper (CPU), beam={args.beam}"
        return partial(asr_faster.transcribe, beam_size=args.beam), log_msg
    from .stages import asr_mlx
    return asr_mlx.transcribe, "ASR backend: mlx (Metal/GPU)"


def _install_drain_handlers(pipeline, log) -> None:
    """1st SIGINT/SIGTERM → graceful drain (finish in-flight, no new files); 2nd → force-quit."""
    def _handler(signum, frame):  # noqa: ARG001 - signal handler signature
        if pipeline.request_drain():
            print(
                "\n[stopping] finishing in-flight files, taking no new ones — "
                "press Ctrl-C again to force-quit.",
                flush=True,
            )
            log.info(f"signal {signum}: graceful drain requested")
        else:
            print("\n[force-quit]", flush=True)
            os._exit(130)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


def cmd_run(cfg: Config, args, log) -> int:
    manifest = Manifest(Path(cfg.systems_folder) / "manifest.json")
    mode = resolve_mode(args)
    opts = build_run_options(args, mode)
    reporter = make_reporter(enabled=not args.no_progress, default_rtf=cfg.progress_default_rtf)
    transcribe, backend_msg = _select_transcribe(cfg, args)
    log.info(backend_msg)
    pipeline = Pipeline(cfg, manifest, transcribe=transcribe, reporter=reporter, console_logs=True)

    try:
        if mode in ("full", "text"):
            tasks = scan_and_hash(Path(cfg.input_folder), manifest, retry_failed=args.retry_failed)
            tasks = filter_tasks(tasks, opts.only, opts.skip)
            log.info(f"{len(tasks)} files to process (mode={mode})")
            total_audio = probe_total_audio(tasks, log)
            reporter.start_batch(total_audio, len(tasks))
            if tasks:
                log.info("preparing models (first run may download several GB)…")
            _install_drain_handlers(pipeline, log)  # Ctrl-C = graceful drain (run_all honors it)
            pipeline.run_all(tasks, opts, jobs=cfg.jobs)
        else:  # summary | resummarize | rerender
            raw_paths = list_raw_files(Path(cfg.systems_folder))
            if opts.only:
                raw_paths = [p for p in raw_paths if opts.only in p.stem]
            log.info(f"{len(raw_paths)} raw files to process (mode={mode})")
            total_audio = sum(_raw_duration(p) for p in raw_paths)
            reporter.start_batch(total_audio, len(raw_paths))
            pipeline.run_existing(raw_paths, opts, jobs=cfg.jobs)
    finally:
        reporter.close()
    return 0


def cmd_enroll(cfg: Config, args, log) -> int:
    """Enroll a name's voiceprint from clean sample audio in the input folder."""
    import tempfile

    from .stages.audio import normalize
    from .stages.diarize import diarize
    from .stages.ingest import scan_audio_files
    from .voiceprints import VoiceprintStore

    files = scan_audio_files(Path(cfg.input_folder))
    if not files:
        print(f"No audio files found in {cfg.input_folder} to enroll from.")
        return 1

    store = VoiceprintStore(Path(cfg.systems_folder) / "voiceprints")
    enrolled = 0
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            wav, _ = normalize(f, Path(tmp))
            diar = diarize(wav, cfg.diarize_device, None, None, None, log)
            if not diar.total_speech_sec:
                print(f"skip {f.name}: no speech detected")
                continue
            dominant = max(diar.total_speech_sec, key=diar.total_speech_sec.get)
            embedding = diar.embeddings.get(dominant)
            if not embedding:
                print(f"skip {f.name}: no embedding for dominant speaker")
                continue
            store.enroll(args.enroll, embedding)
            enrolled += 1
            print(f"enrolled {args.enroll} from {f.name}")

    print(f"Enrolled {enrolled} sample(s) for {args.enroll}.")
    return 0 if enrolled else 1


def _silence_hf_download_bars() -> None:
    """Kill HuggingFace-hub's own tqdm download/reconstruct bars — noise here, and
    they collide with our progress bars (they even show 0.00B when models are
    cached). Must run before mlx-whisper / pyannote import hf. Our own one-line
    'preparing models' heads-up covers a genuine first-run download."""
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # noqa: BLE001 - best-effort; env var already set
        pass


def main(argv: list[str] | None = None) -> int:
    _silence_hf_download_bars()
    # Load HF_TOKEN (and any other vars) from a .env in the working directory so
    # `python -m transcriber` just works after activating the venv, without needing
    # to `source .env` manually. Existing environment variables take precedence.
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))

    args = parse_args(argv)
    cfg = load_config(args.config_path)
    cfg = apply_overrides(cfg, args)
    log = setup_run_logger(Path(cfg.logs_folder), verbose=args.verbose)

    if args.warmup:
        return cmd_warmup(cfg, log)
    if args.enroll:
        return cmd_enroll(cfg, args, log)
    if args.dry_run:
        return cmd_dry_run(cfg, args)
    return cmd_run(cfg, args, log)


if __name__ == "__main__":
    sys.exit(main())
