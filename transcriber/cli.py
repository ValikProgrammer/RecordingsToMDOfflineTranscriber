"""argparse flags + config/CLI merge (§6.2)."""
from __future__ import annotations

import argparse

from .config import Config
from .pipeline import RunOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transcriber", description="Offline transcription + summary pipeline")
    parser.add_argument("--input-folder", "--folder", "--input", dest="input_folder", default=None)
    parser.add_argument("--out", dest="out_folder", default=None)
    parser.add_argument("--config", dest="config_path", default=None)
    parser.add_argument("--only", dest="only", default=None)
    parser.add_argument("--skip", "--exclude", dest="skip", nargs="+", default=None)
    parser.add_argument("--transcribe", "--text", dest="text_mode", action="store_true")
    parser.add_argument("--summary", dest="summary_mode", action="store_true")
    parser.add_argument("--resummarize", dest="resummarize_mode", action="store_true")
    parser.add_argument("--rerender", dest="rerender_mode", action="store_true")
    parser.add_argument("--retry-failed", dest="retry_failed", action="store_true")
    parser.add_argument("--turbo", dest="turbo", action="store_true")
    parser.add_argument("--speakers", dest="speakers", type=int, default=None)
    parser.add_argument("--min-speakers", dest="min_speakers", type=int, default=None)
    parser.add_argument("--max-speakers", dest="max_speakers", type=int, default=None)
    parser.add_argument("--names", dest="names", default=None)
    parser.add_argument("--no-frontmatter", dest="no_frontmatter", action="store_true")
    parser.add_argument("--wikilink-speakers", dest="wikilink_speakers", action="store_true")
    parser.add_argument("--llm-model", dest="llm_model", default=None)
    parser.add_argument("--language", dest="language", default=None)
    parser.add_argument("--jobs", dest="jobs", type=int, default=None)
    parser.add_argument("--diarize-device", dest="diarize_device", choices=["mps", "cpu"], default=None)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--warmup", dest="warmup", action="store_true")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_mode(args: argparse.Namespace) -> str:
    if args.text_mode:
        return "text"
    if args.summary_mode:
        return "summary"
    if args.resummarize_mode:
        return "resummarize"
    if args.rerender_mode:
        return "rerender"
    return "full"


def parse_names(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    names = [n.strip() for n in raw.split(",") if n.strip()]
    return names or None


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.input_folder is not None:
        cfg.input_folder = args.input_folder
    if args.out_folder is not None:
        cfg.out_folder = args.out_folder
    if args.llm_model is not None:
        cfg.llm_model = args.llm_model
    if args.jobs is not None:
        cfg.jobs = args.jobs
    if args.diarize_device is not None:
        cfg.diarize_device = args.diarize_device
    if args.no_frontmatter:
        cfg.obsidian_frontmatter = False
    if args.wikilink_speakers:
        cfg.wikilink_speakers = True
    if args.language is not None:
        cfg.asr_language = args.language
    return cfg


def build_run_options(args: argparse.Namespace, mode: str) -> RunOptions:
    return RunOptions(
        mode=mode,
        only=args.only,
        skip=args.skip,
        turbo=args.turbo,
        speakers=args.speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        names=parse_names(args.names),
        frontmatter=not args.no_frontmatter,
        wikilink_speakers=args.wikilink_speakers,
    )
