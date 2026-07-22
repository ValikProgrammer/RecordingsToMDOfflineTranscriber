import json
import threading
import time
from pathlib import Path

import pytest

from transcriber.config import Config
from transcriber.manifest import Manifest
from transcriber.models import AsrResult, AsrSegment, AsrWord, DiarResult, DiarSegment, FileTask, RawDoc
from transcriber.naming import default_title_for_date, is_technical_name
from transcriber.pipeline import Pipeline, RunOptions, atomic_write_json, filter_tasks, resolve_title_and_date


class _FakeNumpyScalar:
    """Mimics a numpy scalar: not JSON-serializable, but exposes .item()."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def test_atomic_write_json_coerces_numpy_like_scalars(tmp_path):
    path = tmp_path / "raw.json"
    atomic_write_json(path, {"is_monologue": _FakeNumpyScalar(True), "start": _FakeNumpyScalar(1.5)})
    data = json.loads(path.read_text())
    assert data == {"is_monologue": True, "start": 1.5}


def test_atomic_write_json_coerces_real_numpy_types(tmp_path):
    np = pytest.importorskip("numpy")
    path = tmp_path / "raw.json"
    atomic_write_json(
        path,
        {"is_monologue": np.bool_(True), "start": np.float64(1.5), "embedding": np.array([0.1, 0.2])},
    )
    data = json.loads(path.read_text())
    assert data["is_monologue"] is True
    assert data["start"] == 1.5
    assert data["embedding"] == [pytest.approx(0.1), pytest.approx(0.2)]


def test_atomic_write_json_still_rejects_truly_unserializable(tmp_path):
    with pytest.raises(TypeError):
        atomic_write_json(tmp_path / "raw.json", {"bad": object()})


def test_atomic_write_json_leaves_no_temp_file_on_failure(tmp_path):
    with pytest.raises(TypeError):
        atomic_write_json(tmp_path / "raw.json", {"bad": object()})
    # the per-call temp must be cleaned up, not left behind as .*.tmp
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_text_concurrent_same_path_never_interleaves(tmp_path):
    from transcriber.pipeline import atomic_write_text

    path = tmp_path / "out.md"
    a, b = "A" * 50_000, "B" * 50_000
    barrier = threading.Barrier(2)

    def writer(text):
        barrier.wait()
        atomic_write_text(path, text)

    threads = [threading.Thread(target=writer, args=(a,)), threading.Thread(target=writer, args=(b,))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # exactly one writer wins wholesale; bytes are never interleaved and the
    # loser never crashes on a shared temp being replaced out from under it
    assert path.read_text(encoding="utf-8") in (a, b)


def _task(name="team call.m4a", content_hash="blake2b:aaa", status="to_do", path=None) -> FileTask:
    return FileTask(path=path or Path(f"/audio/{name}"), content_hash=content_hash, source_name=name, status=status, reason="")


# --- filter_tasks -------------------------------------------------------

def test_filter_tasks_drops_skip_status():
    tasks = [_task(status="to_do"), _task(content_hash="blake2b:bbb", status="skip")]
    result = filter_tasks(tasks, only=None, skip=None)
    assert [t.status for t in result] == ["to_do"]


def test_filter_tasks_only_matches_stem_or_name():
    tasks = [_task(name="a.m4a", content_hash="blake2b:a"), _task(name="b.m4a", content_hash="blake2b:b")]
    result = filter_tasks(tasks, only="a", skip=None)
    assert [t.source_name for t in result] == ["a.m4a"]


def test_filter_tasks_skip_excludes_by_name():
    tasks = [_task(name="a.m4a", content_hash="blake2b:a"), _task(name="b.m4a", content_hash="blake2b:b")]
    result = filter_tasks(tasks, only=None, skip=["a.m4a"])
    assert [t.source_name for t in result] == ["b.m4a"]


# --- resolve_title_and_date ----------------------------------------------

def test_resolve_title_meaningful_name_ignores_llm_title(tmp_path):
    path = tmp_path / "team call.m4a"
    path.write_bytes(b"x")
    doc = RawDoc(
        schema=1, content_hash="h", source_name=path.name, source_path=str(path),
        language="ru", duration_sec=1.0, num_speakers=1, is_monologue=True,
        asr=None, created_at="", summary=None,
    )
    title, day = resolve_title_and_date(path.name, path, doc)
    assert title == "Team call"


def test_resolve_title_technical_name_uses_llm_title(tmp_path):
    from transcriber.models import AsrInfo, Summary

    path = tmp_path / "2026-07-12.m4a"
    path.write_bytes(b"x")
    assert is_technical_name(path.stem)
    doc = RawDoc(
        schema=1, content_hash="h", source_name=path.name, source_path=str(path),
        language="ru", duration_sec=1.0, num_speakers=1, is_monologue=True,
        asr=AsrInfo("mlx", "large-v3", False), created_at="",
        summary=Summary(title="Title from LLM", text="s"),
    )
    title, day = resolve_title_and_date(path.name, path, doc)
    assert title == "Title from LLM"
    assert day.isoformat() == "2026-07-12"


def test_resolve_title_technical_name_without_summary_falls_back_to_date(tmp_path):
    path = tmp_path / "REC_20260712.m4a"
    path.write_bytes(b"x")
    doc = RawDoc(
        schema=1, content_hash="h", source_name=path.name, source_path=str(path),
        language="ru", duration_sec=1.0, num_speakers=0, is_monologue=True,
        asr=None, created_at="", summary=None,
    )
    title, day = resolve_title_and_date(path.name, path, doc)
    assert title == default_title_for_date(day)


# --- Pipeline.run_all (fresh audio, staged) -------------------------------

def _fake_normalize(path, tmp_dir):
    return Path(str(tmp_dir / f"{path.stem}.wav")), 3.0


def _fake_asr():
    return AsrResult(
        language="ru", backend="mlx", model="large-v3", turbo=False,
        segments=[AsrSegment(0.0, 1.0, "hello", words=[AsrWord("hello", 0.0, 1.0)])],
    )


def _fake_diar():
    return DiarResult(segments=[DiarSegment(0.0, 1.0, "SPEAKER_00")])


def _make_pipeline(tmp_path, **stage_overrides):
    cfg = Config(
        out_folder=str(tmp_path / "out"),
        systems_folder=str(tmp_path / "systems"),
        logs_folder=str(tmp_path / "logs"),
    )
    manifest = Manifest(tmp_path / "systems" / "manifest.json")
    defaults = dict(
        normalize=_fake_normalize,
        transcribe=lambda wav, turbo, log, **kw: _fake_asr(),
        diarize=lambda wav, device, s, mn, mx, log: _fake_diar(),
        summarize=lambda doc, cfg, log: __import__("transcriber.models", fromlist=["Summary"]).Summary(
            title="T", text="Recap"
        ),
    )
    defaults.update(stage_overrides)
    return cfg, manifest, Pipeline(cfg, manifest, **defaults)


def test_run_all_full_mode_marks_done_and_writes_files(tmp_path):
    cfg, manifest, pipeline = _make_pipeline(tmp_path)
    task = _task(path=tmp_path / "team call.m4a")
    task.path.write_bytes(b"fake audio")

    pipeline.run_all([task], RunOptions(mode="full"), jobs=2)

    entry = manifest.get(task.content_hash)
    assert entry.status == "done"
    assert Path(entry.out_path).exists()
    assert Path(entry.raw_path).exists()
    assert "Hello" in Path(entry.out_path).read_text(encoding="utf-8") or "hello" in Path(entry.out_path).read_text(encoding="utf-8")


def test_run_all_text_mode_never_calls_diarize_or_summarize(tmp_path):
    def boom(*a, **k):
        raise AssertionError("should not be called in --text mode")

    cfg, manifest, pipeline = _make_pipeline(tmp_path, diarize=boom, summarize=boom)
    task = _task(path=tmp_path / "team call.m4a")
    task.path.write_bytes(b"fake audio")

    pipeline.run_all([task], RunOptions(mode="text"), jobs=1)

    entry = manifest.get(task.content_hash)
    assert entry.status == "done"
    md = Path(entry.out_path).read_text(encoding="utf-8")
    assert "### Summary" not in md


def test_run_all_stage_a_failure_marks_failed_and_does_not_crash_others(tmp_path):
    def failing_normalize(path, tmp_dir):
        if "bad" in path.name:
            raise RuntimeError("ffmpeg exploded")
        return _fake_normalize(path, tmp_dir)

    cfg, manifest, pipeline = _make_pipeline(tmp_path, normalize=failing_normalize)
    good = _task(name="good.m4a", content_hash="blake2b:good", path=tmp_path / "good.m4a")
    bad = _task(name="bad.m4a", content_hash="blake2b:bad", path=tmp_path / "bad.m4a")
    good.path.write_bytes(b"x")
    bad.path.write_bytes(b"x")

    pipeline.run_all([good, bad], RunOptions(mode="full"), jobs=2)

    assert manifest.get("blake2b:good").status == "done"
    failed = manifest.get("blake2b:bad")
    assert failed.status == "failed"
    assert "ffmpeg exploded" in failed.error


def test_run_all_gpu_stage_is_never_called_concurrently(tmp_path):
    concurrent = {"count": 0, "max": 0}
    lock = threading.Lock()

    def tracking_transcribe(wav, turbo, log, **kw):
        with lock:
            concurrent["count"] += 1
            concurrent["max"] = max(concurrent["max"], concurrent["count"])
        time.sleep(0.05)
        with lock:
            concurrent["count"] -= 1
        return _fake_asr()

    cfg, manifest, pipeline = _make_pipeline(tmp_path, transcribe=tracking_transcribe)
    tasks = []
    for i in range(4):
        p = tmp_path / f"f{i}.m4a"
        p.write_bytes(b"x")
        tasks.append(_task(name=f"f{i}.m4a", content_hash=f"blake2b:h{i}", path=p))

    pipeline.run_all(tasks, RunOptions(mode="full"), jobs=4)

    assert concurrent["max"] == 1
    for t in tasks:
        assert manifest.get(t.content_hash).status == "done"


# --- process_existing_raw (--summary / --resummarize / --rerender) -------

def _write_raw_doc(tmp_path, cfg_systems_folder, content_hash="blake2b:raw1", source_name="team call.m4a"):
    from transcriber.models import AsrInfo, Segment
    from transcriber.pipeline import atomic_write_json, hash_hex

    source_path = tmp_path / source_name
    source_path.write_bytes(b"x")
    doc = RawDoc(
        schema=1, content_hash=content_hash, source_name=source_name, source_path=str(source_path),
        language="ru", duration_sec=10.0, num_speakers=1, is_monologue=True,
        asr=AsrInfo("mlx", "large-v3", False), created_at="2026-01-01T00:00:00Z",
        segments=[Segment(0.0, 2.0, None, "hello")], summary=None,
    )
    raw_path = Path(cfg_systems_folder) / "raw" / f"{hash_hex(content_hash)}.json"
    atomic_write_json(raw_path, doc.to_dict())
    return raw_path, doc


def test_filter_unsummarized_keeps_only_docs_without_summary(tmp_path):
    from transcriber.models import Summary
    from transcriber.pipeline import atomic_write_json, filter_unsummarized, hash_hex

    cfg, _, _ = _make_pipeline(tmp_path)
    no_summary_path, _ = _write_raw_doc(tmp_path, cfg.systems_folder, content_hash="blake2b:pending")
    with_summary_path, doc = _write_raw_doc(tmp_path, cfg.systems_folder, content_hash="blake2b:done")
    doc.summary = Summary(title="T", text="already summarized")
    atomic_write_json(with_summary_path, doc.to_dict())

    result = filter_unsummarized([no_summary_path, with_summary_path])

    assert result == [no_summary_path]


def test_process_existing_raw_resummarize_calls_summarize_not_asr(tmp_path):
    def boom(*a, **k):
        raise AssertionError("ASR must not run for --resummarize")

    cfg, manifest, pipeline = _make_pipeline(tmp_path, normalize=boom, transcribe=boom, diarize=boom)
    raw_path, _ = _write_raw_doc(tmp_path, cfg.systems_folder)

    pipeline.process_existing_raw(raw_path, RunOptions(mode="resummarize"))

    entry = manifest.get("blake2b:raw1")
    assert entry.status == "done"
    md = Path(entry.out_path).read_text(encoding="utf-8")
    assert "Recap" in md


def test_process_existing_raw_rerender_skips_summarize(tmp_path):
    def boom(*a, **k):
        raise AssertionError("summarize must not run for --rerender")

    cfg, manifest, pipeline = _make_pipeline(tmp_path, summarize=boom)
    raw_path, _ = _write_raw_doc(tmp_path, cfg.systems_folder)

    pipeline.process_existing_raw(raw_path, RunOptions(mode="rerender"))

    entry = manifest.get("blake2b:raw1")
    assert entry.status == "done"
    md = Path(entry.out_path).read_text(encoding="utf-8")
    assert "### Summary" not in md


def test_process_existing_raw_reuses_existing_out_path_on_rerender(tmp_path):
    cfg, manifest, pipeline = _make_pipeline(tmp_path)
    raw_path, _ = _write_raw_doc(tmp_path, cfg.systems_folder)

    pipeline.process_existing_raw(raw_path, RunOptions(mode="resummarize"))
    first_out = manifest.get("blake2b:raw1").out_path

    pipeline.process_existing_raw(raw_path, RunOptions(mode="rerender"))
    second_out = manifest.get("blake2b:raw1").out_path

    assert first_out == second_out


def test_stage_b_passes_language_and_prompt_from_config(tmp_path):
    captured = {}

    def spy_transcribe(wav, turbo, log, language=None, initial_prompt=None):
        captured["language"] = language
        captured["initial_prompt"] = initial_prompt
        return _fake_asr()

    cfg, manifest, pipeline = _make_pipeline(tmp_path, transcribe=spy_transcribe)
    cfg.asr_language = "ru"
    cfg.asr_prompt_extra = "ФизТех"
    task = _task(path=tmp_path / "team call.m4a")
    task.path.write_bytes(b"x")

    pipeline.run_all([task], RunOptions(mode="full"), jobs=1)

    assert captured["language"] == "ru"
    assert "ФизТех" in captured["initial_prompt"]


def test_stage_b_auto_language_uses_none_when_detector_undecided(tmp_path):
    captured = {}

    def spy_transcribe(wav, turbo, log, language=None, initial_prompt=None):
        captured["language"] = language
        return _fake_asr()

    def undecided_detect(wav, log, **kw):
        return None  # bilingual/uncertain -> don't force, let the backend switch

    cfg, manifest, pipeline = _make_pipeline(
        tmp_path, transcribe=spy_transcribe, detect_language=undecided_detect
    )
    cfg.asr_language = "auto"
    task = _task(path=tmp_path / "team call.m4a")
    task.path.write_bytes(b"x")

    pipeline.run_all([task], RunOptions(mode="full"), jobs=1)

    assert captured["language"] is None


def test_stage_b_runs_diarize_concurrently_and_merges(tmp_path):
    started = {"asr": threading.Event(), "diar": threading.Event()}
    both_running = {"ok": False}

    def slow_transcribe(wav, turbo, log, **kw):
        started["asr"].set()
        both_running["ok"] = started["diar"].wait(timeout=2.0)
        return _fake_asr()

    def slow_diarize(wav, device, s, mn, mx, log):
        started["diar"].set()
        started["asr"].wait(timeout=2.0)
        return _fake_diar()

    cfg, manifest, pipeline = _make_pipeline(
        tmp_path, transcribe=slow_transcribe, diarize=slow_diarize
    )
    task = _task(path=tmp_path / "team call.m4a")
    task.path.write_bytes(b"x")

    pipeline.run_all([task], RunOptions(mode="full"), jobs=1)

    assert both_running["ok"] is True
    entry = manifest.get(task.content_hash)
    assert entry.status == "done"


def test_stage_b_filters_asr_artifacts(tmp_path):
    def art_transcribe(wav, turbo, log, **kw):
        return AsrResult(
            language="ru", backend="mlx", model="large-v3", turbo=False,
            segments=[
                AsrSegment(0.0, 1.0, "Спасибо за просмотр!", words=[AsrWord("Спасибо", 0.0, 0.5), AsrWord("за", 0.5, 0.7), AsrWord("просмотр", 0.7, 1.0)]),
                AsrSegment(1.0, 2.0, "реальный текст", words=[AsrWord("реальный", 1.0, 1.5), AsrWord("текст", 1.5, 2.0)]),
            ],
        )

    cfg, manifest, pipeline = _make_pipeline(tmp_path, transcribe=art_transcribe)
    task = _task(path=tmp_path / "team call.m4a")
    task.path.write_bytes(b"x")

    pipeline.run_all([task], RunOptions(mode="full"), jobs=1)

    md = Path(manifest.get(task.content_hash).out_path).read_text(encoding="utf-8")
    assert "Спасибо за просмотр" not in md
    assert "реальный текст" in md


def _dv_transcribe(wav, turbo, log, **kw):
    return AsrResult(
        language="ru", backend="mlx", model="large-v3", turbo=False,
        segments=[
            AsrSegment(0.0, 50.0, "aaa", words=[AsrWord("aaa", 0.0, 50.0)]),
            AsrSegment(50.0, 100.0, "bbb", words=[AsrWord("bbb", 50.0, 100.0)]),
        ],
    )


def _dv_diar(wav, device, s, mn, mx, log):
    return DiarResult(
        segments=[DiarSegment(0.0, 50.0, "SPEAKER_00"), DiarSegment(50.0, 100.0, "SPEAKER_01")],
        embeddings={"SPEAKER_00": [1.0, 0.0, 0.0], "SPEAKER_01": [0.0, 1.0, 0.0]},
        total_speech_sec={"SPEAKER_00": 50.0, "SPEAKER_01": 50.0},
    )


def test_stage_b_autonames_speaker_from_voiceprints(tmp_path):
    from transcriber.voiceprints import VoiceprintStore

    VoiceprintStore(tmp_path / "systems" / "voiceprints").enroll("Галя", [1.0, 0.0, 0.0])
    cfg, manifest, pipeline = _make_pipeline(tmp_path, transcribe=_dv_transcribe, diarize=_dv_diar)
    task = _task(path=tmp_path / "team call.m4a")
    task.path.write_bytes(b"x")

    pipeline.run_all([task], RunOptions(mode="full"), jobs=1)

    md = Path(manifest.get(task.content_hash).out_path).read_text(encoding="utf-8")
    assert "Галя" in md


def test_stage_b_enrolls_named_speakers(tmp_path):
    from transcriber.voiceprints import VoiceprintStore

    cfg, manifest, pipeline = _make_pipeline(tmp_path, transcribe=_dv_transcribe, diarize=_dv_diar)
    task = _task(path=tmp_path / "team call.m4a")
    task.path.write_bytes(b"x")

    pipeline.run_all([task], RunOptions(mode="full", names=["Галя", "Иван"]), jobs=1)

    store = VoiceprintStore(tmp_path / "systems" / "voiceprints")
    assert store.identify([1.0, 0.0, 0.0], threshold=0.7) == "Галя"
    assert store.identify([0.0, 1.0, 0.0], threshold=0.7) == "Иван"


def test_pretty_skips_when_output_exists_unless_forced(tmp_path):
    calls = {"n": 0}

    def spy_pretty(doc, cfg, log):
        calls["n"] += 1
        return "PRETTY BODY"

    cfg, _, pipeline = _make_pipeline(tmp_path, pretty_transcript=spy_pretty)
    raw_path, _ = _write_raw_doc(tmp_path, cfg.systems_folder)

    # first pass: pretty file doesn't exist yet -> generated
    pipeline.process_existing_raw(raw_path, RunOptions(mode="rerender", pretty=True))
    assert calls["n"] == 1

    # second pass: pretty file exists -> skipped, not regenerated
    pipeline.process_existing_raw(raw_path, RunOptions(mode="rerender", pretty=True))
    assert calls["n"] == 1

    # with --force: regenerated even though the file exists
    pipeline.process_existing_raw(raw_path, RunOptions(mode="rerender", pretty=True, force=True))
    assert calls["n"] == 2


def test_pretty_flag_writes_pretty_file(tmp_path):
    cfg, manifest, pipeline = _make_pipeline(tmp_path, pretty_transcript=lambda doc, cfg, log: "PRETTY BODY")
    task = _task(path=tmp_path / "team call.m4a")
    task.path.write_bytes(b"x")

    pipeline.run_all([task], RunOptions(mode="full", pretty=True), jobs=1)

    out_path = Path(manifest.get(task.content_hash).out_path)
    pretty_path = tmp_path / "out" / "pretty" / out_path.name
    assert pretty_path.exists()
    pretty = pretty_path.read_text(encoding="utf-8")
    # pretty is now a full document: frontmatter + summary + the rewritten transcript
    assert pretty.startswith("---\n")
    assert "### Summary" in pretty
    assert "### Transcript" in pretty
    assert "PRETTY BODY" in pretty
    # the verbatim segment dump is replaced, not appended alongside the pretty body
    assert "**[00:00]" not in pretty


def test_request_drain_first_true_then_false(tmp_path):
    _, _, pipeline = _make_pipeline(tmp_path)
    assert pipeline.request_drain() is True   # first signal: start draining
    assert pipeline.request_drain() is False  # second signal: caller should force-quit


def test_drain_stops_taking_new_files(tmp_path):
    # After drain is requested, stage A must not pull any new files (run still exits cleanly).
    processed = []

    def spy_transcribe(wav, turbo, log, **kw):
        processed.append(str(wav))
        return _fake_asr()

    cfg, manifest, pipeline = _make_pipeline(tmp_path, transcribe=spy_transcribe)
    pipeline.request_drain()  # drain before the run even starts
    tasks = []
    for i in range(4):
        t = _task(name=f"f{i}.m4a", content_hash=f"blake2b:{i}", path=tmp_path / f"f{i}.m4a")
        t.path.write_bytes(b"x")
        tasks.append(t)

    pipeline.run_all(tasks, RunOptions(mode="text"), jobs=2)  # returns cleanly, takes nothing new

    assert processed == []
    assert all(manifest.get(t.content_hash) is None for t in tasks)


def test_stage_a_does_not_run_far_ahead_of_gpu(tmp_path):
    # Regression: stage A must not eagerly churn the whole batch (memory blowup).
    # With stage B blocked, stage A should stall after ~lookahead files, not all of them.
    import time

    release = threading.Event()
    starts = []
    lock = threading.Lock()

    def counting_normalize(path, tmp_dir):
        with lock:
            starts.append(path.name)
        return _fake_normalize(path, tmp_dir)

    def blocking_transcribe(wav, turbo, log, **kw):
        release.wait(timeout=10)  # stall the GPU stage
        return _fake_asr()

    cfg, manifest, pipeline = _make_pipeline(
        tmp_path, normalize=counting_normalize, transcribe=blocking_transcribe
    )
    cfg.stage_a_lookahead = 3
    tasks = []
    for i in range(12):
        t = _task(name=f"f{i}.m4a", content_hash=f"blake2b:{i}", path=tmp_path / f"f{i}.m4a")
        t.path.write_bytes(b"x")
        tasks.append(t)

    th = threading.Thread(target=pipeline.run_all, args=(tasks, RunOptions(mode="text"), 2))
    th.start()
    time.sleep(0.6)  # let stage A run as far ahead as it's allowed
    with lock:
        ahead = len(starts)
    release.set()
    th.join(timeout=15)

    # lookahead(3) + stage_a_out queue(jobs=2) + in-flight slack — never the whole batch of 12
    assert ahead <= 6, f"stage A ran {ahead} files ahead — backpressure not working"


def test_auto_language_forces_detector_result(tmp_path):
    captured = {}

    def spy_transcribe(wav, turbo, log, **kw):
        captured["language"] = kw.get("language")
        return _fake_asr()

    def fake_detect(wav, log, **kw):
        captured["detect_called"] = True
        return "en"

    cfg, manifest, pipeline = _make_pipeline(tmp_path, transcribe=spy_transcribe, detect_language=fake_detect)
    cfg.asr_language = "auto"
    task = _task(path=tmp_path / "call.m4a")
    task.path.write_bytes(b"x")

    pipeline.run_all([task], RunOptions(mode="text"), jobs=1)

    assert captured["detect_called"] is True
    assert captured["language"] == "en"  # detected language is forced on transcribe


def test_forced_language_skips_detector(tmp_path):
    captured = {}

    def spy_transcribe(wav, turbo, log, **kw):
        captured["language"] = kw.get("language")
        return _fake_asr()

    def boom_detect(wav, log, **kw):
        raise AssertionError("detector must not run when language is forced")

    cfg, manifest, pipeline = _make_pipeline(tmp_path, transcribe=spy_transcribe, detect_language=boom_detect)
    cfg.asr_language = "ru"
    task = _task(path=tmp_path / "call.m4a")
    task.path.write_bytes(b"x")

    pipeline.run_all([task], RunOptions(mode="text"), jobs=1)

    assert captured["language"] == "ru"


def test_no_pretty_flag_skips_pretty_file(tmp_path):
    called = {"n": 0}

    def spy(doc, cfg, log):
        called["n"] += 1
        return "x"

    cfg, manifest, pipeline = _make_pipeline(tmp_path, pretty_transcript=spy)
    task = _task(path=tmp_path / "team call.m4a")
    task.path.write_bytes(b"x")

    pipeline.run_all([task], RunOptions(mode="full"), jobs=1)

    assert called["n"] == 0
    assert not (tmp_path / "out" / "pretty").exists()
