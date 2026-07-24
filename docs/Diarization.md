# Diarization (`--diarize`) + voiceprint enrollment

Speaker labels on transcripts, as their own stage in the multi-stage manifest.
Standalone `--diarize` is a **post-pass**: it attaches speakers to an existing
text raw (no ASR). Sample-based mono pre-check skips full-file pyannote on
solo recordings.

## Stages (manifest)

Each file tracks `text` / `diarize` / `summary` / `pretty`. Legacy
`status=done` migrates to **`text=done` only**; other stages start `pending`.

## `--diarize` (post-pass)

Needs a completed `text` stage and `systems/raw/<hash>.json`. Uses the source
audio again (normalize → mono pre-check → pyannote → merge into the raw →
re-render markdown). Does **not** re-run Whisper.

```bash
# speakers on already-transcribed files
python -m transcriber --diarize --input-folder ./audio

# optional names + voiceprint enroll
python -m transcriber --diarize --names "мама,Арина" --input-folder ./audio
```

| Situation | Manifest |
|-----------|----------|
| No raw / `text` not done | `diarize=skipped`, reason `no_transcript` |
| Mono pre-check: one voice | `diarize=skipped`, reason `mono` (no full pyannote) |
| Multi / pre-check unsure | full diarize → `diarize=done` |

`--force` redoes `done` / `skipped` diarize stages.

## Transcript + speakers in one shot

```bash
python -m transcriber --text --diarize --input-folder ./audio
# or full pipeline (also summarizes):
python -m transcriber --input-folder ./audio
```

ASR and diarize run in parallel on the same wav (after mono pre-check).

## Timecodes vs speakers

**Timecodes** (`start`/`end` on segments) come from ASR. Diarization only
answers *who* spoke; merge attaches labels to existing segments.

## Then summary / pretty

```bash
python -m transcriber --summary
python -m transcriber --pretty --summary
```

Those modes gate on **manifest stages**, not on empty raw fields / file
existence.

## Voiceprints

Embeddings live on `speakers_meta[*].embedding` after a successful diarize.

### `--enroll NAME`

Dominant speaker of each sample in the input folder → store under `NAME`.

### `--enroll-raw PATH|NAME`

Enroll every named speaker from a labeled raw (no audio / no pyannote).

```bash
python -m transcriber --enroll-raw "dialog with mama"
python -m transcriber --enroll-raw systems/raw/ab12cd34.json
```

## Code map

- `transcriber/cli.py` — `--diarize`, `want_diarize`, modes
- `transcriber/pipeline.py` — `run_diarize_pass`, stage updates, `filter_need_stage`
- `transcriber/stages/mono_precheck.py` — sample windows → mono decision
- `transcriber/stages/diarize.py` — pyannote
- `transcriber/voiceprints.py` — enroll / identify
- `transcriber/manifest.py` — schema v2 stages + legacy migration

Design: `docs/superpowers/specs/2026-07-22-multi-stage-manifest-and-post-diarize-design.md`
