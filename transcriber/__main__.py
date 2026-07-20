"""Entry point: python -m transcriber (§6, §16 milestones 1 & 10)."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from .cli import apply_overrides, build_run_options, parse_args, resolve_mode
from .config import Config, load_config
from .logging_setup import setup_run_logger
from .manifest import Manifest
from .pipeline import Pipeline, filter_tasks, list_raw_files
from .stages.audio import FfmpegNotFoundError, check_ffmpeg_available, normalize
from .stages.ingest import scan_and_hash


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


def cmd_run(cfg: Config, args, log) -> int:
    manifest = Manifest(Path(cfg.systems_folder) / "manifest.json")
    mode = resolve_mode(args)
    opts = build_run_options(args, mode)
    pipeline = Pipeline(cfg, manifest)

    if mode in ("full", "text"):
        tasks = scan_and_hash(Path(cfg.input_folder), manifest, retry_failed=args.retry_failed)
        tasks = filter_tasks(tasks, opts.only, opts.skip)
        log.info(f"{len(tasks)} files to process (mode={mode})")
        pipeline.run_all(tasks, opts, jobs=cfg.jobs)
    else:  # summary | resummarize | rerender
        raw_paths = list_raw_files(Path(cfg.systems_folder))
        if opts.only:
            raw_paths = [p for p in raw_paths if opts.only in p.stem]
        log.info(f"{len(raw_paths)} raw files to process (mode={mode})")
        pipeline.run_existing(raw_paths, opts, jobs=cfg.jobs)
    return 0


def main(argv: list[str] | None = None) -> int:
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
    if args.dry_run:
        return cmd_dry_run(cfg, args)
    return cmd_run(cfg, args, log)


if __name__ == "__main__":
    sys.exit(main())
