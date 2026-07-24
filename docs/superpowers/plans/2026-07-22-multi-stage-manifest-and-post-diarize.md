# Multi-stage manifest + post-pass diarize — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the manifest the single source of truth for `text` / `diarize` / `summary` / `pretty` stages, and turn `--diarize` into a post-pass that adds speakers onto existing transcripts (with sample-based mono skip).

**Architecture:** Extend `ManifestEntry` with a `stages` map (schema v2). Migrate legacy `status=done` → `text=done` only. Gate `scan_and_hash`, `--summary`, and `--pretty` on stage status. Split “want diarize” from mode: standalone `--diarize` = post-pass on raw; `--text --diarize` / `full` = ASR∥diarize on wav. Mono pre-check samples windows before full-file pyannote.

**Tech Stack:** Python 3.x, existing pyannote wrapper, pytest, argparse CLI.

**Spec:** `docs/superpowers/specs/2026-07-22-multi-stage-manifest-and-post-diarize-design.md`

**GitHub issue:** create/link when `gh` auth works (draft body in plan Task 0). Closes #35 (mono pre-check).

## Global Constraints

- Manifest is the only gate for whether a stage runs (not `summary is None`, not pretty file existence, not root `status=done`).
- Legacy migration: old `done` → **only** `text=done`; `diarize`/`summary`/`pretty` → `pending` (no disk inference).
- Standalone `--diarize` does **not** run ASR; skips with `reason=no_transcript` if text missing.
- Mono pre-check failure → fallback to full-file diarize.
- Timecodes come from ASR; diarize only attaches speakers.
- Keep `--enroll` / `--enroll-raw` behavior (read speakers from raw).
- Run tests with `.venv/bin/python -m pytest`.
- Commit after each task; do not push unless asked.
- Include already-uncommitted diarize/enroll-raw WIP in Task 0 foundation commit(s) on a dedicated branch.

## File map

| File | Role |
|------|------|
| `transcriber/models.py` | `StageStatus`, `ManifestEntry.stages` |
| `transcriber/manifest.py` | schema v2, migrate on load |
| `transcriber/stages/ingest.py` | stage-aware `scan_and_hash` |
| `transcriber/stages/mono_precheck.py` | **create** — sample windows, decide mono vs multi |
| `transcriber/pipeline.py` | stage updates; post-pass diarize; joint flags; summary/pretty gates |
| `transcriber/cli.py` | `RunOptions.want_diarize`; mode semantics |
| `transcriber/__main__.py` | route post-pass vs fresh-audio |
| `docs/Diarization.md`, `docs/Summary.md`, `docs/prettier.md` | user docs |
| tests under `tests/` | migration, ingest, pipeline, cli, mono_precheck |

---

### Task 0: Branch, issue draft, commit foundation WIP

**Files:**
- Create branch `feat/multi-stage-manifest-diarize` from current HEAD (includes dirty tree)
- Create: `docs/superpowers/plans/2026-07-22-multi-stage-manifest-and-post-diarize.md` (this file — already being written)
- Spec already at `docs/superpowers/specs/2026-07-22-multi-stage-manifest-and-post-diarize-design.md`
- Commit existing diarize-mode + enroll-raw + docs WIP as foundation

**Interfaces:**
- Produces: clean branch with prior `--diarize`/`--enroll-raw` code committed so later tasks rewrite semantics on a known base.

- [ ] **Step 1: Create branch**

```bash
git checkout -b feat/multi-stage-manifest-diarize
```

- [ ] **Step 2: Write issue body file** (gh auth may be broken — still write the body)

Create `docs/superpowers/issues/2026-07-22-multi-stage-manifest-diarize.md`:

```markdown
## Summary
Manifest tracks per-stage status (text/diarize/summary/pretty). `--diarize` becomes a post-pass on existing raws. Sample mono pre-check skips full pyannote (closes #35).

## Spec
docs/superpowers/specs/2026-07-22-multi-stage-manifest-and-post-diarize-design.md

## Acceptance
- [ ] Legacy `done` migrates to `text=done` only
- [ ] `--diarize` on text-only raws adds speakers without re-ASR
- [ ] No raw / no text → skip `no_transcript`
- [ ] Mono pre-check → `diarize=skipped` reason `mono`
- [ ] `--summary`/`--pretty` gate on manifest stages
- [ ] `--text --diarize` runs ASR∥diarize together

## Closes
#35
```

- [ ] **Step 3: Try `gh issue create`**; if auth fails, leave body file and continue

```bash
gh issue create --title "Multi-stage manifest + post-pass diarize" --body-file docs/superpowers/issues/2026-07-22-multi-stage-manifest-diarize.md
```

- [ ] **Step 4: Commit foundation WIP** (diarize mode, enroll-raw, specs, Diarization.md, related tests — exclude unrelated noise like Benchmarks if not needed; include diarization-related files)

```bash
git add \
  docs/Diarization.md \
  docs/superpowers/specs/2026-07-22-diarize-mode-and-enroll-raw-design.md \
  docs/superpowers/specs/2026-07-22-multi-stage-manifest-and-post-diarize-design.md \
  docs/superpowers/plans/2026-07-22-multi-stage-manifest-and-post-diarize.md \
  docs/superpowers/issues/2026-07-22-multi-stage-manifest-diarize.md \
  transcriber/cli.py transcriber/pipeline.py transcriber/__main__.py transcriber/voiceprints.py \
  tests/test_cli.py tests/test_pipeline.py tests/test_voiceprints.py
git status
git commit -m "$(cat <<'EOF'
feat: diarize mode + enroll-raw foundation for multi-stage work

Land speaker-only mode and enroll-from-raw before rewriting --diarize
into a post-pass gated by per-stage manifest status.
EOF
)"
```

If other modified files (`summarize.py`, `pretty.py`, etc.) are required for green tests on this branch, include them in the same commit or a follow-up `fix:` commit — do not leave the suite red.

- [ ] **Step 5: Verify foundation tests**

Run: `.venv/bin/python -m pytest tests/test_cli.py tests/test_pipeline.py tests/test_voiceprints.py tests/test_manifest.py -q`
Expected: PASS

---

### Task 1: `StageStatus` + `ManifestEntry.stages` + migration

**Files:**
- Modify: `transcriber/models.py` (`ManifestEntry`)
- Modify: `transcriber/manifest.py` (`SCHEMA_VERSION`, `_load` migration)
- Test: `tests/test_manifest.py`

**Interfaces:**
- Produces:
  - `STAGE_NAMES = ("text", "diarize", "summary", "pretty")`
  - `@dataclass StageState: status: str; updated_at: str = ""; reason: str | None = None`
  - `ManifestEntry.stages: dict[str, StageState]`
  - `default_stages() -> dict[str, StageState]` all `pending`
  - `migrate_entry(raw_dict) -> ManifestEntry` / migrate inside `_load`
  - Legacy: root `status=="done"` and no stages → `text=done`, others `pending`
  - `SCHEMA_VERSION = 2`
  - Helpers: `stage_status(entry, name) -> str`, `set_stage(entry, name, status, reason=None) -> ManifestEntry` (or methods on Manifest)

- [ ] **Step 1: Failing tests** in `tests/test_manifest.py`:

```python
def test_migrate_legacy_done_sets_only_text_done(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"schema": 1, "entries": {"blake2b:aaa": {'
        '"content_hash": "blake2b:aaa", "source_name": "a.m4a", "status": "done",'
        '"updated_at": "2026-01-01T00:00:00Z"}}}'
    )
    from transcriber.manifest import Manifest
    m = Manifest(path)
    e = m.get("blake2b:aaa")
    assert e.stages["text"].status == "done"
    assert e.stages["diarize"].status == "pending"
    assert e.stages["summary"].status == "pending"
    assert e.stages["pretty"].status == "pending"


def test_migrate_does_not_infer_summary_from_elsewhere(tmp_path):
    # even if we later have raw with summary, migration alone must leave summary pending
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"schema": 1, "entries": {"blake2b:aaa": {'
        '"content_hash": "blake2b:aaa", "source_name": "a.m4a", "status": "done"}}}'
    )
    from transcriber.manifest import Manifest
    e = Manifest(path).get("blake2b:aaa")
    assert e.stages["summary"].status == "pending"


def test_new_entry_has_all_stages_pending(tmp_path):
    from transcriber.manifest import Manifest
    from transcriber.models import ManifestEntry, default_stages
    m = Manifest(tmp_path / "manifest.json")
    m.upsert(ManifestEntry(content_hash="blake2b:x", source_name="x.m4a", status="in_progress", stages=default_stages()))
    e = Manifest(tmp_path / "manifest.json").get("blake2b:x")
    assert all(e.stages[s].status == "pending" for s in ("text", "diarize", "summary", "pretty"))
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `.venv/bin/python -m pytest tests/test_manifest.py::test_migrate_legacy_done_sets_only_text_done -v`

- [ ] **Step 3: Implement models + migration**

In `models.py` add `StageState`, `default_stages()`, extend `ManifestEntry` with `stages: dict = field(default_factory=default_stages)`.

In `manifest.py`: `SCHEMA_VERSION = 2`. On load, if entry lacks `stages` or schema<2, build stages per migration table in the spec. Persist `stages` via `asdict` (StageState must be a dataclass). When loading dict stages, reconstruct `StageState(**…)`.

Preserve unknown root fields. When upserting, always write schema 2.

- [ ] **Step 4: Tests PASS** + update any tests that construct `ManifestEntry` without stages (defaults should cover).

Run: `.venv/bin/python -m pytest tests/test_manifest.py tests/stages/test_ingest.py -q`

- [ ] **Step 5: Commit**

```bash
git add transcriber/models.py transcriber/manifest.py tests/test_manifest.py
git commit -m "feat: manifest schema v2 with per-stage status + legacy migration"
```

---

### Task 2: Stage-aware `scan_and_hash`

**Files:**
- Modify: `transcriber/stages/ingest.py`
- Test: `tests/stages/test_ingest.py`

**Interfaces:**
- Change signature to:
  `scan_and_hash(folder, manifest, *, retry_failed=False, need_stage: str | None = None, force: bool = False) -> list[FileTask]`
- When `need_stage` is None: preserve backward-compatible behavior **except** treat “fully skip” as: skip only if caller didn’t ask for a stage (legacy paths). Prefer: **always pass `need_stage` from pipeline**.
- Rules when `need_stage="text"`:
  - no entry → `to_do`
  - `stages["text"].status == "done"` and not force → `skip` reason `text already done`
  - `failed` root + not retry → skip as today
  - `in_progress` / pending text → `redo`/`to_do`
- When `need_stage="diarize"` (audio still required for post-pass):
  - skip if `diarize` in (`done`,`skipped`) and not force
  - still return `to_do`/`redo` even if `text` done (post-pass needs the file path)
- Root `status=="done"` alone must **not** skip when `need_stage` is set and that stage is pending.

- [ ] **Step 1: Failing test**

```python
def test_scan_skips_on_stage_done_not_root_done(tmp_path):
    # entry with text done, diarize pending, root status leftover "done"
    ...
    tasks = scan_and_hash(folder, manifest, need_stage="diarize")
    assert tasks[0].status == "to_do"  # or redo — runnable


def test_scan_skips_diarize_when_stage_done(tmp_path):
    ...
    tasks = scan_and_hash(folder, manifest, need_stage="diarize")
    assert tasks[0].status == "skip"
```

- [ ] **Step 2: Implement + fix callers** in `__main__.py` / pipeline to pass `need_stage`.

- [ ] **Step 3: pytest** `tests/stages/test_ingest.py` PASS

- [ ] **Step 4: Commit** `feat: stage-aware scan_and_hash`

---

### Task 3: CLI — `want_diarize` + mode semantics

**Files:**
- Modify: `transcriber/cli.py` (`resolve_mode`, `RunOptions` via `pipeline.RunOptions`, `build_run_options`)
- Modify: `transcriber/pipeline.py` (`RunOptions` dataclass)
- Test: `tests/test_cli.py`

**Interfaces:**
- `RunOptions.want_diarize: bool`
- `resolve_mode`:
  - `--diarize` alone (no `--text`) → `"diarize"` (post-pass)
  - `--text` → `"text"` (ASR); if also `--diarize`, still `"text"` but `want_diarize=True`
  - default/`full` → `"full"` with `want_diarize=True`
  - `--summary` / etc. unchanged
- `build_run_options`:
  - `want_diarize = args.diarize_mode or mode == "full"`
  - Note: `--text --diarize` → mode text, want_diarize True

- [ ] **Step 1: Tests**

```python
def test_text_plus_diarize_sets_want_diarize():
    args = parse_args(["--text", "--diarize"])
    assert resolve_mode(args) == "text"
    opts = build_run_options(args, "text")
    assert opts.want_diarize is True

def test_diarize_alone_is_postpass_mode():
    assert resolve_mode(parse_args(["--diarize"])) == "diarize"

def test_full_wants_diarize():
    opts = build_run_options(parse_args([]), "full")
    assert opts.want_diarize is True
```

Update existing `test_resolve_mode_diarize` if it assumed fresh-audio-only semantics — keep mode name `"diarize"`, change pipeline behavior in Task 4–5.

- [ ] **Step 2: Implement**

- [ ] **Step 3: Commit** `feat: CLI want_diarize; --text --diarize combines`

---

### Task 4: Pipeline writes stage statuses (text / full)

**Files:**
- Modify: `transcriber/pipeline.py` (`_mark_in_progress`, success upsert, `_fail`, `_safe_stage_b` branch on `want_diarize`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- On successful `--text` without diarize: `stages.text=done`; do not set diarize done.
- On successful text+diarize or full audio path with diarize: `text=done` and `diarize=done` (or skipped mono — Task 5).
- On full with summary: `summary=done`.
- On pretty written: `pretty=done`; if skipped by force logic later — Task 6.
- `_mark_in_progress` / `_fail` must **preserve** existing `stages` from prior entry (merge, don’t wipe to defaults).
- `_safe_stage_b`: if `opts.mode == "text"` and not `opts.want_diarize` → ASR only; if want_diarize → ASR∥diarize path (same as today’s non-text branch).

- [ ] **Step 1: Tests** asserting manifest stages after `run_all` fakes.

- [ ] **Step 2: Implement helpers** e.g. `_with_stage(entry, name, status, reason=None)` 

- [ ] **Step 3: Commit** `feat: pipeline records per-stage manifest status`

---

### Task 5: Mono pre-check module

**Files:**
- Create: `transcriber/stages/mono_precheck.py`
- Test: `tests/stages/test_mono_precheck.py`

**Interfaces:**
- `WINDOW_SEC = 30` (reuse langdetect window spacing via `langdetect.window_starts` or copy)
- `SHORT_FULL_SEC = 300`  # files shorter than this → one window covering whole file (configurable constant OK)
- `def is_likely_monologue(wav: Path, duration: float, diarize_fn, device: str, log) -> bool | None`
  - Returns `True` (mono → skip full), `False` (run full), `None` (error → caller falls back to full)
  - Implementation: build list of window starts; for each window extract via ffmpeg to temp wav; call `diarize_fn(window_wav, device, None, None, None, log)`; collect speaker counts from `total_speech_sec` / segments; if every successful window has ≤1 speaker → `True`; if any window has ≥2 → `False`; if all windows fail → `None`
- Pure helpers for decision from speaker-count list: `decide_mono(counts: list[int]) -> bool | None`

- [ ] **Step 1: Unit-test `decide_mono`** without pyannote

```python
assert decide_mono([1, 1, 1]) is True
assert decide_mono([1, 2, 1]) is False
assert decide_mono([]) is None
```

- [ ] **Step 2: Implement module**

- [ ] **Step 3: Commit** `feat: sample-based mono pre-check for diarize`

---

### Task 6: Post-pass `--diarize` + wire mono pre-check

**Files:**
- Modify: `transcriber/pipeline.py` — new `run_diarize_pass` (or extend `run_existing`)
- Modify: `transcriber/__main__.py` — mode `"diarize"` → post-pass, not `run_all` ASR path
- Test: `tests/test_pipeline.py`

**Interfaces:**
- For each audio task from `scan_and_hash(..., need_stage="diarize")`:
  1. Load raw by hash under `systems/raw/{hash}.json`. Missing or `text` stage not done → set `diarize=skipped` reason `no_transcript`, log, continue.
  2. Normalize wav → mono pre-check:
     - mono True → `diarize=skipped` reason `mono`; do not call full diarize
     - False/None → `self.diarize` full file → `merge` into existing ASR segments (reuse merge with doc’s asr reconstruction **or** merge(asr_from_doc, diar) — implement `merge_stage.merge` against segments already on doc: prefer building a lightweight AsrResult from doc.segments + calling existing `merge`)
  3. Voiceprints if enabled; write raw+md; `diarize=done`
- `--force` redoes skipped/done.
- Joint path in `_safe_stage_b`: before full diarize, run mono pre-check; on mono, build text doc then optionally strip/skip speakers (same as text-only speakers) and mark `diarize=skipped` reason `mono`.

- [ ] **Step 1: Tests with fakes** for no_transcript, mono skip (spy diarize not called), multi merge.

- [ ] **Step 2: Implement**

- [ ] **Step 3: Update/remove obsolete test** `test_run_all_diarize_mode_diarizes_but_skips_summary` — replace with post-pass tests + joint text+diarize test.

- [ ] **Step 4: Commit** `feat: post-pass --diarize with mono pre-check`

---

### Task 7: Summary + pretty gate on manifest

**Files:**
- Modify: `transcriber/pipeline.py` — replace `filter_unsummarized`; pretty skip
- Modify: `transcriber/__main__.py` — pass manifest into filter
- Test: `tests/test_pipeline.py`, `tests/stages/test_summarize.py` if needed
- Docs: `docs/Summary.md`, `docs/prettier.md`

**Interfaces:**
- `filter_need_stage(manifest, raw_paths, stage: str, force: bool) -> list[Path]`
  - Resolve hash from raw filename stem / doc.content_hash; include if stage pending or force
  - Ensure manifest entry exists (create pending stages if raw exists but no entry — edge case: create entry with `text=done`, others pending, target stage pending)
- Pretty: if `stages.pretty` done and not force → skip; else write and set `pretty=done`
- After summary write: `summary=done`

- [ ] **Step 1: Tests**

- [ ] **Step 2: Implement + doc updates**

- [ ] **Step 3: Commit** `feat: summary/pretty gated by manifest stages`

---

### Task 8: Docs + Diarization.md rewrite + final suite

**Files:**
- Modify: `docs/Diarization.md` — post-pass semantics, mono skip, multi-stage
- Modify: `myReadme.md` only if it documents `--diarize` wrongly
- Spec superseded banner already on old design doc

- [ ] **Step 1: Rewrite Diarization.md** to match spec

- [ ] **Step 2: Full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 3: Commit** `docs: multi-stage manifest and post-pass diarize`

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| stages map + statuses + reason | 1 |
| migrate done → text only | 1 |
| stage-aware skip | 2 |
| `--text --diarize` parallel | 3, 4, 6 |
| post-pass `--diarize` | 6 |
| no_transcript skip | 6 |
| mono pre-check B + fallback | 5, 6 |
| summary/pretty from manifest | 7 |
| docs | 7, 8 |
| closes #35 | 5, 6 + issue |
| enroll-raw unchanged | 0 foundation |

## Execution

User asked to run immediately: use **subagent-driven-development** after Task 0 is done in-session (branch + foundation commit + issue attempt).
