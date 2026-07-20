import re
from pathlib import Path

from transcriber.logging_setup import per_file_log_path, setup_file_logger, setup_run_logger

LOG_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z \[INFO\] .+$")


def test_per_file_log_path_matches_spec_example(tmp_path):
    path = per_file_log_path(tmp_path, "рабочая встреча.m4a", "a1b2c3d4")
    assert path == tmp_path / "rabochaya-vstrecha__a1b2c3d4.log"


def test_setup_run_logger_writes_utc_formatted_line(tmp_path):
    logger = setup_run_logger(tmp_path)
    logger.info("taken: рабочая встреча.m4a (hash a1b2c3d4)")
    for handler in logger.handlers:
        handler.flush()
    content = (tmp_path / "run.log").read_text(encoding="utf-8").strip()
    assert LOG_LINE_RE.match(content)
    assert "taken: рабочая встреча.m4a" in content


def test_setup_file_logger_writes_to_expected_path(tmp_path):
    log_path = per_file_log_path(tmp_path, "рабочая встреча.m4a", "a1b2c3d4")
    logger = setup_file_logger(log_path)
    logger.info("ASR done: language=ru, segments=142")
    for handler in logger.handlers:
        handler.flush()
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8").strip()
    assert LOG_LINE_RE.match(content)
    assert "ASR done: language=ru, segments=142" in content


def test_setup_file_logger_does_not_duplicate_handlers_on_reuse(tmp_path):
    log_path = tmp_path / "x__aaaa.log"
    logger1 = setup_file_logger(log_path)
    logger2 = setup_file_logger(log_path)
    logger2.info("only once")
    for handler in logger2.handlers:
        handler.flush()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
