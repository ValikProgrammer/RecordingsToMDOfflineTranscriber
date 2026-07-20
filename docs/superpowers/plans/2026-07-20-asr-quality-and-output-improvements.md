# ASR Quality & Output Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix garbage transcription (wrong language, hallucinations, mangled terms), make per-record topics complete and timecode-first, and attempt a safe speedup — all without lowering transcription quality.

**Architecture:** The ASR stage (`asr_mlx.transcribe`) currently calls `mlx_whisper.transcribe` with no language/prompt/anti-hallucination options. We force the decode language (config, `ru` default), pass anti-hallucination knobs, and add a configurable RU/IT glossary prompt. The pipeline passes these from config. Rendering and the summary prompt change to make topics complete and timecode-first. Finally a best-effort, measured overlap of diarization with ASR.

**Tech Stack:** Python 3.14, `mlx_whisper`, `pyannote.audio` 4.x, Ollama (qwen2.5), pytest.

## Global Constraints

- Change only these files (+ their tests): `transcriber/config.py`, `config.example.toml`, `transcriber/cli.py`, `transcriber/stages/asr_mlx.py`, `transcriber/pipeline.py`, `transcriber/stages/summarize.py`, `transcriber/stages/render.py`.
- Quality first: never trade transcription accuracy for speed. `--turbo` stays opt-in; large-v3 remains the default.
- `mlx_whisper.transcribe` is imported lazily inside `transcribe()` (module must stay importable without MLX).
- Run tests with `.venv/bin/python -m pytest`.
- Backward compatibility: `transcribe()` gains keyword params with defaults; existing callers/fakes must keep working.

---

### Task 1: Config fields `asr_language` and `asr_prompt_extra`

**Files:**
- Modify: `transcriber/config.py` (Config dataclass ~line 43-52; `_SIMPLE_KEYS` line 15-19)
- Modify: `config.example.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config.asr_language: str` (default `"ru"`), `Config.asr_prompt_extra: str` (default `""`), both loadable from TOML top-level keys.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_load_config_reads_asr_language_and_prompt_extra(tmp_path, monkeypatch):
    from transcriber.config import load_config

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('asr_language = "en"\nasr_prompt_extra = "ФизТех, Богодаров"\n')
    monkeypatch.chdir(tmp_path)

    cfg = load_config(None)

    assert cfg.asr_language == "en"
    assert cfg.asr_prompt_extra == "ФизТех, Богодаров"


def test_config_defaults_asr_language_ru():
    from transcriber.config import Config

    assert Config().asr_language == "ru"
    assert Config().asr_prompt_extra == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py::test_config_defaults_asr_language_ru -v`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'asr_language'`

- [ ] **Step 3: Add fields and TOML keys**

In `transcriber/config.py`, add to the `Config` dataclass (next to `asr_model`):

```python
    asr_language: str = "ru"
    asr_prompt_extra: str = ""
```

And extend `_SIMPLE_KEYS` to include the two new keys:

```python
_SIMPLE_KEYS = (
    "input_folder", "out_folder", "systems_folder", "logs_folder",
    "asr_model", "asr_language", "asr_prompt_extra",
    "llm_model", "llm_ctx", "diarize_device",
    "mono_threshold", "jobs", "obsidian_frontmatter", "wikilink_speakers",
)
```

- [ ] **Step 4: Document in config.example.toml**

Add under the `asr_model` line in `config.example.toml`:

```toml
asr_language = "ru"          # forced decode language; "" or "auto" = auto-detect
asr_prompt_extra = ""        # extra terms/names added to the ASR glossary prompt, e.g. "ФизТех, Иван Богодаров"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add transcriber/config.py config.example.toml tests/test_config.py
git commit -m "Add asr_language and asr_prompt_extra config fields"
```

---

### Task 2: CLI `--language` override

**Files:**
- Modify: `transcriber/cli.py` (`build_parser`, `apply_overrides`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Config.asr_language` (Task 1).
- Produces: `--language` flag → `args.language`; `apply_overrides` sets `cfg.asr_language` when provided.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_language_flag_overrides_asr_language():
    from transcriber.cli import apply_overrides, parse_args
    from transcriber.config import Config

    cfg = apply_overrides(Config(), parse_args(["--language", "en"]))
    assert cfg.asr_language == "en"


def test_no_language_flag_keeps_config_asr_language():
    from transcriber.cli import apply_overrides, parse_args
    from transcriber.config import Config

    cfg = Config(asr_language="ru")
    cfg = apply_overrides(cfg, parse_args([]))
    assert cfg.asr_language == "ru"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli.py::test_language_flag_overrides_asr_language -v`
Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'language'`

- [ ] **Step 3: Add the flag and override**

In `transcriber/cli.py` `build_parser()`, after the `--llm-model` line:

```python
    parser.add_argument("--language", dest="language", default=None)
```

In `apply_overrides()`, before `return cfg`:

```python
    if args.language is not None:
        cfg.asr_language = args.language
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add transcriber/cli.py tests/test_cli.py
git commit -m "Add --language CLI flag to override ASR decode language"
```

---

### Task 3: ASR glossary prompt + anti-hallucination knobs in `transcribe()`

**Files:**
- Modify: `transcriber/stages/asr_mlx.py`
- Test: `tests/stages/test_asr_mlx.py`

**Interfaces:**
- Produces:
  - `DEFAULT_INITIAL_PROMPT: str` — built-in RU/IT glossary sentence.
  - `build_initial_prompt(extra: str) -> str` — returns the default prompt, with `extra` appended (space-joined, stripped) when non-empty.
  - `transcribe(wav, turbo, log, language=None, initial_prompt=None) -> AsrResult` — passes `language` (only when truthy), `initial_prompt`, `condition_on_previous_text=False`, `hallucination_silence_threshold=2.0`, `word_timestamps=True` to `mlx_whisper.transcribe`.

- [ ] **Step 1: Write the failing test**

The existing `tests/stages/test_asr_mlx.py` already injects a fake `mlx_whisper` (check how it does so and reuse that mechanism). Add tests that capture kwargs. If the file installs the fake via `monkeypatch.setitem(sys.modules, "mlx_whisper", fake)`, follow the same pattern:

```python
import sys
import types
import logging

LOG = logging.getLogger("test")


def _install_fake_mlx(monkeypatch, captured):
    fake = types.SimpleNamespace()

    def fake_transcribe(audio, **kwargs):
        captured.update(kwargs)
        captured["audio"] = audio
        return {"language": "ru", "segments": []}

    fake.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)


def test_transcribe_passes_language_and_antihallucination(monkeypatch, tmp_path):
    from transcriber.stages.asr_mlx import transcribe

    captured = {}
    _install_fake_mlx(monkeypatch, captured)
    transcribe(tmp_path / "a.wav", turbo=False, log=LOG, language="ru", initial_prompt="glossary here")

    assert captured["language"] == "ru"
    assert captured["condition_on_previous_text"] is False
    assert captured["hallucination_silence_threshold"] == 2.0
    assert captured["initial_prompt"] == "glossary here"
    assert captured["word_timestamps"] is True


def test_transcribe_omits_language_when_auto(monkeypatch, tmp_path):
    from transcriber.stages.asr_mlx import transcribe

    captured = {}
    _install_fake_mlx(monkeypatch, captured)
    transcribe(tmp_path / "a.wav", turbo=False, log=LOG, language=None, initial_prompt=None)

    assert "language" not in captured


def test_build_initial_prompt_appends_extra():
    from transcriber.stages.asr_mlx import DEFAULT_INITIAL_PROMPT, build_initial_prompt

    assert build_initial_prompt("") == DEFAULT_INITIAL_PROMPT
    combined = build_initial_prompt("ФизТех, Богодаров")
    assert combined.startswith(DEFAULT_INITIAL_PROMPT)
    assert "ФизТех, Богодаров" in combined
```

> NOTE for the implementer: open `tests/stages/test_asr_mlx.py` first and match its existing fake-injection style rather than assuming the helper above. Keep existing tests passing.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/stages/test_asr_mlx.py::test_build_initial_prompt_appends_extra -v`
Expected: FAIL with `ImportError: cannot import name 'build_initial_prompt'`

- [ ] **Step 3: Implement**

In `transcriber/stages/asr_mlx.py`, add after the repo constants:

```python
DEFAULT_INITIAL_PROMPT = (
    "Совещание. Обсуждаем ФизТех, хакатон, стипендию, ментора, практику, "
    "дедлайн, проект, репозиторий, коммит, деплой, бэкенд, фронтенд, API, "
    "Телеграм-бота, субботнюю школу."
)


def build_initial_prompt(extra: str) -> str:
    extra = (extra or "").strip()
    return f"{DEFAULT_INITIAL_PROMPT} {extra}".strip() if extra else DEFAULT_INITIAL_PROMPT
```

Change the `transcribe` signature and the `mlx_whisper.transcribe` call:

```python
def transcribe(
    wav: Path,
    turbo: bool,
    log: logging.Logger,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> AsrResult:
    import mlx_whisper

    repo = TURBO_REPO if turbo else FULL_REPO
    log.info(f"ASR start: model={repo}")
    kwargs = dict(
        path_or_hf_repo=repo,
        word_timestamps=True,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
        initial_prompt=initial_prompt,
    )
    if language:
        kwargs["language"] = language
    raw = mlx_whisper.transcribe(str(wav), **kwargs)
    segments = [_convert_segment(s) for s in raw["segments"]]
    language_detected = raw.get("language", "unknown")
    log.info(f"ASR done: language={language_detected}, segments={len(segments)}")
    return AsrResult(
        language=language_detected,
        segments=segments,
        backend="mlx",
        model="large-v3-turbo" if turbo else "large-v3",
        turbo=turbo,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/stages/test_asr_mlx.py -v`
Expected: PASS (all, including pre-existing tests)

- [ ] **Step 5: Commit**

```bash
git add transcriber/stages/asr_mlx.py tests/stages/test_asr_mlx.py
git commit -m "Force language, disable hallucination, add glossary prompt in ASR"
```

---

### Task 4: Wire pipeline to pass language + composed prompt into `transcribe()`

**Files:**
- Modify: `transcriber/pipeline.py` (`_safe_stage_b`, ~line 228-236)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Config.asr_language`, `Config.asr_prompt_extra` (Task 1); `asr_mlx.build_initial_prompt` (Task 3); `transcribe(..., language=, initial_prompt=)` (Task 3).
- Produces: stage B calls `self.transcribe(wav, turbo, log, language=<normalized>, initial_prompt=<composed>)` where language is `None` when `cfg.asr_language` is `""`/`"auto"` (case-insensitive), else `cfg.asr_language`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
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


def test_stage_b_auto_language_passes_none(tmp_path):
    captured = {}

    def spy_transcribe(wav, turbo, log, language=None, initial_prompt=None):
        captured["language"] = language
        return _fake_asr()

    cfg, manifest, pipeline = _make_pipeline(tmp_path, transcribe=spy_transcribe)
    cfg.asr_language = "auto"
    task = _task(path=tmp_path / "team call.m4a")
    task.path.write_bytes(b"x")

    pipeline.run_all([task], RunOptions(mode="full"), jobs=1)

    assert captured["language"] is None
```

Also update the default fake in `_make_pipeline` so it accepts the new kwargs (change the `transcribe` default lambda):

```python
        transcribe=lambda wav, turbo, log, **kw: _fake_asr(),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::test_stage_b_passes_language_and_prompt_from_config -v`
Expected: FAIL (`language`/`initial_prompt` not passed → captured empty / None)

- [ ] **Step 3: Implement**

In `transcriber/pipeline.py`, add the import near the other stage imports:

```python
from .stages.asr_mlx import build_initial_prompt
```

In `_safe_stage_b`, replace the ASR call line:

```python
            lang = self.cfg.asr_language.strip()
            language = None if lang.lower() in ("", "auto") else lang
            prompt = build_initial_prompt(self.cfg.asr_prompt_extra)
            asr = self.transcribe(ctx.wav_path, opts.turbo, log, language=language, initial_prompt=prompt)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add transcriber/pipeline.py tests/test_pipeline.py
git commit -m "Pass configured language and glossary prompt into ASR stage"
```

---

### Task 5: Topics rendering — timecode-first, one per line, uncapped

**Files:**
- Modify: `transcriber/stages/render.py` (lines 83-87)
- Test: `tests/stages/test_render.py`

**Interfaces:**
- Consumes: `doc.summary.topics` — list of `TopicRef(term: str, ts: float)`.
- Produces: a `**Topics:**` header followed by one `- [MM:SS] term` line per topic, in order, with no cap.

- [ ] **Step 1: Write the failing test**

Add to `tests/stages/test_render.py` (follow the file's existing helpers for building a `RawDoc`/`Summary`; the assertions below are the contract):

```python
def test_topics_render_one_per_line_timecode_first():
    from transcriber.models import TopicRef

    topics = [TopicRef(term=f"Topic {i}", ts=float(i * 60)) for i in range(30)]
    md = _render_with_topics(topics)  # helper: builds a RawDoc+Summary with these topics and calls render_markdown

    assert "**Topics:**" in md
    # timecode first, dash bullet, one line each, all 30 present
    assert "- [00:00] Topic 0" in md
    assert "- [29:00] Topic 29" in md
    assert md.count("\n- [") >= 30
    # old joined format must be gone
    assert " · Topic" not in md
```

> NOTE: if `test_render.py` has no `_render_with_topics` helper, write one at the top of the test using the same `RawDoc`/`Summary` construction the other tests in that file already use. Reuse existing fixtures; do not invent new model fields.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/stages/test_render.py::test_topics_render_one_per_line_timecode_first -v`
Expected: FAIL (current output is a single `·`-joined line, `- [00:00]` absent)

- [ ] **Step 3: Implement**

In `transcriber/stages/render.py`, replace the topics block (currently lines 83-87):

```python
        if doc.summary.topics:
            parts.append("**Topics:**")
            for t in doc.summary.topics:
                parts.append(f"- [{format_timecode(t.ts, use_hours)}] {t.term}")
            parts.append("")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/stages/test_render.py -v`
Expected: PASS (all; fix any pre-existing render test that asserted the old `·` topics format)

- [ ] **Step 5: Commit**

```bash
git add transcriber/stages/render.py tests/stages/test_render.py
git commit -m "Render topics one per line, timecode first, uncapped"
```

---

### Task 6: Summary prompt — extract topics proportional to content; reduce dedupes only duplicates

**Files:**
- Modify: `transcriber/stages/summarize.py` (`SYSTEM_PROMPT_TEMPLATE`; `_map_reduce` reduce prompt ~line 118-125)
- Test: `tests/stages/test_summarize.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: system prompt text contains a topics-coverage instruction; reduce prompt instructs dedupe of only identical/near-identical topics (not collapsing distinct ones).

- [ ] **Step 1: Write the failing test**

Add to `tests/stages/test_summarize.py`:

```python
def test_system_prompt_instructs_full_topic_coverage():
    from transcriber.stages.summarize import SYSTEM_PROMPT_TEMPLATE

    prompt = SYSTEM_PROMPT_TEMPLATE.format(language="ru", sentences="5-8", long_form_hint="")
    lowered = prompt.lower()
    assert "topic" in lowered
    # must ask for one topic per distinct subject and no cap
    assert "distinct" in lowered
    assert "do not limit" in lowered or "no limit" in lowered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/stages/test_summarize.py::test_system_prompt_instructs_full_topic_coverage -v`
Expected: FAIL (`"do not limit"` not present)

- [ ] **Step 3: Implement**

In `transcriber/stages/summarize.py`, add a line to `SYSTEM_PROMPT_TEMPLATE` just before the closing `"""` (after the `{long_form_hint}` line):

```python
Extract one "topics" entry per distinct subject discussed, in chronological order.
Do not limit the number of topics — a long recording with many subjects should have many topics. Do not pad with trivial or duplicate topics.
```

In `_map_reduce`, change the reduce prompt string so it only dedupes true duplicates:

```python
    reduce_prompt = (
        "Combine these partial summaries (JSON list below) into ONE final JSON object "
        "with the same schema. Keep ALL distinct topics in chronological order; "
        "merge only topics that are duplicates or clearly the same subject. "
        "Do not drop distinct topics or cap their number:\n"
        + json.dumps(partials, ensure_ascii=False)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/stages/test_summarize.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add transcriber/stages/summarize.py tests/stages/test_summarize.py
git commit -m "Prompt for complete, chronological, uncapped topics"
```

---

### Task 7: Best-effort diarize/ASR overlap in stage B (measured)

**Files:**
- Modify: `transcriber/pipeline.py` (`_safe_stage_b`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `self.transcribe`, `self.diarize`, `self.merge` (unchanged signatures).
- Produces: identical merged `ctx.doc` as before; diarization runs on a worker thread concurrently with ASR, then results are merged after both finish.

- [ ] **Step 1: Write the failing test (proves correctness + concurrency wiring)**

Add to `tests/test_pipeline.py`:

```python
def test_stage_b_runs_diarize_concurrently_and_merges(tmp_path):
    import threading

    started = {"asr": threading.Event(), "diar": threading.Event()}
    both_running = {"ok": False}

    def slow_transcribe(wav, turbo, log, **kw):
        started["asr"].set()
        # confirm diarize is allowed to start before ASR returns
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

    assert both_running["ok"] is True  # diarize started while ASR was still running
    entry = manifest.get(task.content_hash)
    assert entry.status == "done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::test_stage_b_runs_diarize_concurrently_and_merges -v`
Expected: FAIL (`both_running["ok"]` is False — diarize currently starts only after ASR returns)

- [ ] **Step 3: Implement concurrent diarization**

In `transcriber/pipeline.py`, add near the top imports (if not already present):

```python
from concurrent.futures import ThreadPoolExecutor
```

(`ThreadPoolExecutor` is already imported for `run_all`; reuse it — do not add a duplicate import.)

Rewrite the non-text branch of `_safe_stage_b` so diarization starts on a thread before ASR, then both join before merge:

```python
    def _safe_stage_b(self, ctx: _Ctx, opts: RunOptions) -> _Ctx | None:
        task, log = ctx.task, ctx.log
        try:
            lang = self.cfg.asr_language.strip()
            language = None if lang.lower() in ("", "auto") else lang
            prompt = build_initial_prompt(self.cfg.asr_prompt_extra)

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
                )
            ctx.doc = doc
            log.info(f"ASR done: language={doc.language}, segments={len(doc.segments)}")
            return ctx
        except Exception as exc:  # noqa: BLE001
            self._fail(task, ctx.log_path, log, exc)
            return None
```

> This supersedes the ASR-call edit from Task 4 (the language/prompt wiring is folded in here). Keep Task 4's tests green.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: PASS (all, including Task 4 tests)

- [ ] **Step 5: Measure on a real file (keep or revert)**

Restore the test audio into `./audio/`, then:

```bash
.venv/bin/python -m transcriber --input-folder ./audio --out ./out --retry-failed -v
```

Read `logs/<slug>.log`: compare the wall-clock between `ffmpeg` line and `merge done` line against the sequential baseline (~152s for the 22-min file). If the overlap is NOT faster (e.g. GPU contention makes ASR+diarize slower than sequential), `git revert` this task's commit and note the result in the plan. If faster, keep it.

- [ ] **Step 6: Commit**

```bash
git add transcriber/pipeline.py tests/test_pipeline.py
git commit -m "Overlap diarization with ASR in stage B (best-effort)"
```

---

### Task 8: Full-suite verification + real acceptance run

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all)

- [ ] **Step 2: Real acceptance**

With `./audio/Natalia talk about scheduler.m4a` restored:

```bash
.venv/bin/python -m transcriber --input-folder ./audio --out ./out --retry-failed -v
```

Open the generated `out/*.md` and confirm:
- Header shows `Language: RU` (not NN).
- No `Thank you for watching` / `Takk for watching` block.
- `ФизТех` is recognized (not "FDI").
- `Topics` is a timecode-first bulleted list, one per line.

If any check fails, that is a new debugging cycle (use systematic-debugging), not a plan step.

---

## Self-Review

**Spec coverage:**
- A (language) → Tasks 1, 2, 4/7. A (anti-hallucination) → Task 3. A (glossary prompt + extra) → Tasks 1, 3, 4. ✓
- B (safe speedup, measured) → Task 7 (with measure/revert step). ✓
- C (topics timecode-first, one per line, uncapped) → Task 5. C (summary produces enough topics) → Task 6. ✓
- Testing + acceptance → Task 8. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. Test helpers that depend on existing fixtures carry an explicit NOTE to reuse the file's existing style (render/asr_mlx). ✓

**Type consistency:** `transcribe(wav, turbo, log, language=None, initial_prompt=None)` used identically in Tasks 3, 4, 7. `build_initial_prompt(extra: str) -> str` defined in Task 3, imported in Task 4/7. `TopicRef(term, ts)` matches models. Config fields `asr_language`/`asr_prompt_extra` consistent across Tasks 1, 2, 4, 7. ✓

**Note:** Task 4 edits the ASR call; Task 7 rewrites the same method and folds Task 4's wiring in. Implementer must keep Task 4's tests green after Task 7 (called out inline).
