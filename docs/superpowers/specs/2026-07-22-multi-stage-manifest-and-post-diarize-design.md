# Multi-stage manifest + post-pass diarize — design

Manifest is the single source of truth for which pipeline stages have run.
`--diarize` becomes a pass that adds speakers onto an existing transcript raw
(no ASR). A sample-based monologue pre-check skips full pyannote on solo
recordings. Supersedes the “fresh-audio only / cannot diarize later” limitation
in [2026-07-22-diarize-mode-and-enroll-raw-design.md](./2026-07-22-diarize-mode-and-enroll-raw-design.md)
(that doc’s Part 1 mode meaning changes; Part 2 `--enroll-raw` stays).

## Motivation

- ~30 files already transcribed as `--text` need speakers without re-running ASR.
- Today `manifest.status=done` means “finished whatever mode last ran,” so a
  text-only file blocks `--diarize`, and there is no per-stage visibility.
- Summary/pretty already support later passes, but they gate on raw fields /
  file existence — inconsistent with a stage-aware manifest.
- Timecodes (`start`/`end` on segments) come from ASR, not diarization.
  Diarization only answers *who* spoke; merge attaches labels to existing
  segments.

## Goals

1. Per-stage status in the manifest: `text`, `diarize`, `summary`, `pretty`.
2. `--diarize` alone: find raw with completed text, run diarize (+ mono
   pre-check), write speakers back, re-render markdown. No transcript → skip.
3. `--text` together with diarize (or `full` audio portion): ASR and diarize in
   parallel on the same wav, as today.
4. Mono pre-check (option B): sample windows; one voice → skip full pyannote,
   mark `diarize=skipped` with reason `mono`.
5. Migrate legacy `status=done` → **only** `text=done`; all other stages
   `pending` (no inference from existing raw/pretty artifacts).
6. `--summary` / `--pretty` decide work from manifest stages, not from
   `summary is None` / pretty file existence.

Non-goals: changing ASR quality, voiceprint math, or `--enroll` /
`--enroll-raw` behavior (except that they keep reading speakers from raw).

## Stage model

| Stage | “done” means | Needs audio? |
|-------|----------------|--------------|
| `text` | Raw exists with ASR segments (text + timecodes) | yes |
| `diarize` | Speakers applied to segments, **or** intentional mono skip | yes (to decide / run) |
| `summary` | Summary written into raw | no |
| `pretty` | Pretty markdown artifact written | no |

Per-stage status values: `pending` | `in_progress` | `done` | `skipped` | `failed`.

Each stage object:

```json
{ "status": "done", "updated_at": "2026-07-22T12:00:00Z" }
```

For `skipped` / `failed`, optional `reason` (e.g. `no_transcript`, `mono`,
short error text).

Root `ManifestEntry.status` remains a coarse batch/UI flag
(`in_progress` / `failed` / similar). It must **not** gate “already fully
processed.” Skip/run decisions use `stages` only.

## Manifest schema (v2)

`SCHEMA_VERSION`: 1 → **2**.

Example entry:

```json
{
  "content_hash": "blake2b:…",
  "source_name": "call.m4a",
  "status": "in_progress",
  "stages": {
    "text":    { "status": "done", "updated_at": "…" },
    "diarize": { "status": "pending" },
    "summary": { "status": "pending" },
    "pretty":  { "status": "pending" }
  },
  "language": "ru",
  "num_speakers": null,
  "duration_sec": 120.5,
  "out_path": "…",
  "raw_path": "…",
  "log_path": "…",
  "error": null,
  "updated_at": "…"
}
```

Raw JSON and pretty files still hold content. The manifest alone decides whether
a stage should run.

### Migration on load

When loading a v1 entry (no `stages`):

| Legacy root `status` | Migration |
|----------------------|-----------|
| `done` | `text=done` with `updated_at`; `diarize` / `summary` / `pretty` = `pending` |
| `failed` | all stages `pending`; keep root `failed` + `error` |
| `in_progress` | all stages `pending` |
| already has `stages` | leave unchanged |

**Do not** set `summary`/`pretty`/`diarize` to `done` by inspecting raw or disk.
User-approved rule: old `done` meant text only; everything else starts pending.
Consequence: files that already have summary/pretty may be processed again on
the next `--summary` / `--pretty` until those stages are marked done — accepted.

Bump schema to 2 after migration write-back (lazy on load + save, or one-shot
upgrade — implementer’s choice as long as every read path sees `stages`).

## CLI / mode behavior

### `--text`

ASR only → raw + verbatim md (no speakers). Set `text=done`. Leave other
stages untouched.

### `--diarize` (new meaning: post-pass)

Does **not** run ASR. For each candidate (input-folder audio keyed by content
hash, reconciled with manifest + raw):

1. No raw, or `text` ≠ `done` → set `diarize=skipped`, `reason=no_transcript`;
   log clearly; continue.
2. `diarize` already `done` or `skipped` → skip unless `--force`.
3. Else: resolve source audio → normalize → mono pre-check (below):
   - mono → do not run full-file pyannote; set `diarize=skipped`, `reason=mono`;
     log that diarize was skipped as monologue. (Segments stay without
     multi-speaker labels; re-render optional/no-op if nothing changed.)
   - multi / uncertain → full diarize → merge into **existing** raw → voiceprints
     if enabled → re-render verbatim md → `diarize=done`.
4. `--names` / voiceprint enroll-on-confirm: unchanged, after a successful
   full diarize.

### `--text` + diarize together / `full` audio portion

Single job: normalize → ASR ∥ diarize (subject to mono pre-check) → merge →
write raw. Update `text` and `diarize` in one pass. Same parallelism as today’s
stage B.

### `full`

Audio stages as above, then summary (and pretty if `--pretty`). Mark each stage
as it finishes.

### `--summary` / `--pretty`

Gate on manifest only:

- `--summary`: run where `text=done` and `summary=pending` (or always if
  `--force`); on success `summary=done`.
- `--pretty`: run where `pretty=pending` (need a usable raw / out path); on
  success `pretty=done`.
- `--force`: redo even if stage is `done` or `skipped`.

Replace `filter_unsummarized` (raw `summary is None`) and pretty “file exists”
checks with manifest stage queries. Still write content into raw / `out/pretty/`.

### Compatibility note

Previously `--diarize` meant “fresh audio: ASR + speakers, no summary”
([diarize-mode design](./2026-07-22-diarize-mode-and-enroll-raw-design.md)).
New meaning: speakers-only pass on existing text. “Transcript + speakers in one
shot” = `--text` with diarize enabled in the same invocation, or `full`.
Update `docs/Diarization.md` and CLI help accordingly.

## Mono pre-check (closes issue #35)

Same sampling idea as lang-detect: up to three windows (start / middle / end).
Short files: feed the whole clip (config threshold, e.g. duration below N
seconds or a few minutes → single window covering the file).

Run a **lightweight** multi-speaker check on those windows only (reuse
diarize/embedding stack on samples — not a full-file pyannote pass).

| Pre-check result | Action | Manifest |
|------------------|--------|----------|
| One voice | Skip full-file pyannote | `diarize=skipped`, `reason=mono` |
| ≥2 voices or uncertain | Full-file diarize + merge | `diarize=done` on success |
| Pre-check errors | **Fallback: run full-file diarize** (safe default) | then `done` / `failed` as usual |

`--diarize --force` re-evaluates even prior `skipped` mono.

Exact detector API (pyannote on windows vs embedding distance across windows)
can be chosen in the implementation plan; this spec requires sample-based
behavior, skip-on-mono, and fallback-to-full on pre-check failure.

## Code map (expected touch points)

| Area | Change |
|------|--------|
| `transcriber/models.py` | `ManifestEntry.stages`; stage status dataclass/dict shape |
| `transcriber/manifest.py` | schema v2; migrate on load |
| `transcriber/stages/ingest.py` | `scan_and_hash` / task selection by requested stage(s), not root `done` |
| `transcriber/pipeline.py` | post-pass diarize; joint text+diarize; stage updates; remove raw/file gates for summary/pretty |
| `transcriber/cli.py`, `__main__.py` | mode wiring; help text |
| new helper (e.g. under `stages/`) | mono pre-check sampling + decide |
| `docs/Diarization.md`, `Summary.md`, `prettier.md`, user-facing readme | document new semantics |
| old diarize-mode spec | banner: Part 1 superseded by this doc |

## Testing

Minimum:

- Migration: legacy `done` → `text=done`, other stages `pending`.
- `--diarize` without raw / without `text=done` → `skipped` / `no_transcript`.
- `--diarize` with raw + mono pre-check → pyannote not called; `skipped` / `mono`.
- `--diarize` with raw + multi → merge, `diarize=done`, markdown has speakers.
- Combined text + diarize → both stages `done`; ASR and diarize both invoked.
- `--summary` selects via manifest `pending`, not raw field.
- `--pretty` selects via manifest `pending`, not file existence.
- `--force` re-runs `done` / `skipped` stages.

## Rollout / process

1. This design approved in-repo.
2. GitHub issue covering: multi-stage manifest, post-pass `--diarize`, mono
   pre-check (closes #35), summary/pretty gate migration, docs.
3. Implementation plan + PR against the issue.
4. After merge: run `--diarize` on the folder of ~30 text-only files (migrated
   manifest shows `diarize=pending`).

## Decisions (locked)

| Topic | Decision |
|-------|----------|
| Source of truth for stages | Manifest only (approach 1) |
| Legacy `done` migration | Text only; all other stages `pending` |
| Standalone `--diarize` | Post-pass on existing raw; skip if no transcript |
| Joint text + diarize | Parallel on wav |
| Solo recordings | Sample pre-check → skip full pyannote; `skipped`/`mono` |
| Summary / pretty gate | Manifest stages |
| Timecodes | From ASR, not diarize |
| Pre-check failure | Fallback to full diarize |
