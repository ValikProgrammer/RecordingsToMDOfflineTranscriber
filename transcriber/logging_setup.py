"""Run log + per-file log setup with UTC timestamps (§5.3)."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from .naming import slugify_for_log


class UtcFormatter(logging.Formatter):
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", self.converter(record.created))
        return f"{ts}Z [{record.levelname}] {record.getMessage()}"


class TqdmLoggingHandler(logging.Handler):
    """Emit records via ``tqdm.write`` so live progress bars aren't corrupted.

    Writes to the same stream tqdm draws its bars on (stderr). When no bars are
    active ``tqdm.write`` just prints, so this is safe to use unconditionally.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm

            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:  # pragma: no cover - never let logging kill the run
            self.handleError(record)


def setup_run_logger(logs_folder: Path, verbose: bool = False) -> logging.Logger:
    logs_folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("transcriber.run")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(logs_folder / "run.log", encoding="utf-8")
    fh.setFormatter(UtcFormatter())
    logger.addHandler(fh)
    sh = TqdmLoggingHandler()
    sh.setFormatter(UtcFormatter())
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def per_file_log_path(logs_folder: Path, source_name: str, hash8: str) -> Path:
    stem = slugify_for_log(Path(source_name).stem)
    return logs_folder / f"{stem}__{hash8}.log"


def setup_file_logger(log_path: Path, verbose: bool = False, console: bool = False) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"transcriber.file.{log_path.stem}")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(UtcFormatter())
    logger.addHandler(fh)
    if console:  # surface each file's stage internals on the console (always-verbose)
        ch = TqdmLoggingHandler()
        ch.setFormatter(UtcFormatter())
        logger.addHandler(ch)
    logger.propagate = False
    return logger
