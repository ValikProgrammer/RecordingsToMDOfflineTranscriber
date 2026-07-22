import subprocess
from pathlib import Path

from transcriber.__main__ import cmd_dry_run, cmd_run, cmd_warmup, main
from transcriber.cli import parse_args
from transcriber.config import Config
from transcriber.logging_setup import setup_run_logger
from transcriber.manifest import Manifest
from transcriber.models import ManifestEntry
from transcriber.stages.ingest import compute_blake2b


def _cfg(tmp_path, **overrides) -> Config:
    cfg = Config(
        input_folder=str(tmp_path / "audio"),
        out_folder=str(tmp_path / "out"),
        systems_folder=str(tmp_path / "systems"),
        logs_folder=str(tmp_path / "logs"),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    Path(cfg.input_folder).mkdir(parents=True, exist_ok=True)
    return cfg


def _make_m4a(path: Path, duration_sec: int = 1) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", str(duration_sec), "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )


def test_cmd_dry_run_reports_new_file_without_processing(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    _make_m4a(Path(cfg.input_folder) / "test.m4a")

    exit_code = cmd_dry_run(cfg, parse_args([]))

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "to_do" in out
    assert "test.m4a" in out


def test_cmd_dry_run_respects_retry_failed_flag(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    audio_path = Path(cfg.input_folder) / "test.m4a"
    _make_m4a(audio_path)
    manifest = Manifest(Path(cfg.systems_folder) / "manifest.json")
    manifest.upsert(ManifestEntry(content_hash=compute_blake2b(audio_path), source_name="test.m4a", status="failed", error="boom"))

    without_retry = cmd_dry_run(cfg, parse_args([]))
    out = capsys.readouterr().out
    assert without_retry == 0
    assert "skip" in out

    with_retry = cmd_dry_run(cfg, parse_args(["--retry-failed"]))
    out = capsys.readouterr().out
    assert with_retry == 0
    assert "redo" in out


def test_cmd_dry_run_empty_folder_prints_message(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    exit_code = cmd_dry_run(cfg, parse_args([]))
    assert exit_code == 0
    assert "No supported audio files found" in capsys.readouterr().out


def test_cmd_run_full_mode_fails_gracefully_when_a_dependency_is_unavailable(tmp_path, monkeypatch):
    # Whichever heavy dependency is missing (mlx-whisper if setup_mac.sh hasn't
    # run yet, or HF_TOKEN if pyannote's gated model isn't authorized), the run
    # must mark the file failed with a real error instead of crashing.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg = _cfg(tmp_path)
    _make_m4a(Path(cfg.input_folder) / "test.m4a")
    log = setup_run_logger(Path(cfg.logs_folder))
    args = parse_args(["--input-folder", cfg.input_folder])

    cmd_run(cfg, args, log)

    manifest = Manifest(Path(cfg.systems_folder) / "manifest.json")
    entries = list(manifest.all_entries().values())
    assert len(entries) == 1
    assert entries[0].status == "failed"
    assert entries[0].error


def _write_raw(cfg, content_hash, summary):
    from transcriber.models import AsrInfo, RawDoc, Segment
    from transcriber.pipeline import atomic_write_json, hash_hex

    doc = RawDoc(
        schema=1, content_hash=content_hash, source_name=f"{content_hash}.m4a", source_path="x",
        language="ru", duration_sec=1.0, num_speakers=1, is_monologue=True,
        asr=AsrInfo("mlx", "large-v3", False), created_at="2026-01-01T00:00:00Z",
        segments=[Segment(0.0, 1.0, None, "hi")], summary=summary,
    )
    atomic_write_json(Path(cfg.systems_folder) / "raw" / f"{hash_hex(content_hash)}.json", doc.to_dict())


def test_cmd_run_summary_skips_already_summarized_unless_forced(tmp_path, monkeypatch):
    from transcriber.models import Summary

    cfg = _cfg(tmp_path)
    _write_raw(cfg, "blake2b:pending", None)
    _write_raw(cfg, "blake2b:done", Summary(title="T", text="already summarized"))

    captured = {}
    monkeypatch.setattr(
        "transcriber.__main__.Pipeline.run_existing",
        lambda self, raw_paths, opts, jobs: captured.__setitem__("paths", list(raw_paths)),
    )
    log = setup_run_logger(Path(cfg.logs_folder))

    cmd_run(cfg, parse_args(["--summary"]), log)
    assert len(captured["paths"]) == 1  # only the unsummarized raw doc

    captured.clear()

    cmd_run(cfg, parse_args(["--summary", "--force"]), log)
    assert len(captured["paths"]) == 2


def test_cmd_warmup_reports_a_clear_error_when_a_dependency_is_unavailable(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg = _cfg(tmp_path)
    log = setup_run_logger(Path(cfg.logs_folder))
    exit_code = cmd_warmup(cfg, log)
    assert exit_code == 1
    assert capsys.readouterr().out.strip()


def test_main_dry_run_end_to_end(tmp_path, capsys, monkeypatch):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _make_m4a(audio_dir / "test.m4a")
    monkeypatch.chdir(tmp_path)

    exit_code = main([
        "--dry-run",
        "--input-folder", str(audio_dir),
        "--out", str(tmp_path / "out"),
    ])

    assert exit_code == 0
    assert "test.m4a" in capsys.readouterr().out


def test_main_loads_hf_token_from_dotenv(tmp_path, monkeypatch):
    import os

    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / "audio").mkdir()
    (tmp_path / ".env").write_text("HF_TOKEN=token-from-dotenv\n")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--dry-run", "--input-folder", str(tmp_path / "audio")])

    assert exit_code == 0
    assert os.environ.get("HF_TOKEN") == "token-from-dotenv"


def test_main_does_not_override_existing_hf_token(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("HF_TOKEN", "token-from-shell")
    (tmp_path / "audio").mkdir()
    (tmp_path / ".env").write_text("HF_TOKEN=token-from-dotenv\n")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--dry-run", "--input-folder", str(tmp_path / "audio")])

    assert exit_code == 0
    assert os.environ.get("HF_TOKEN") == "token-from-shell"
