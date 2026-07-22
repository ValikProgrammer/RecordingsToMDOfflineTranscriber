# Diarize mode + enroll-from-raw — design

> **Superseded (Part 1):** standalone `--diarize` meaning and “cannot diarize
> later” are replaced by
> [2026-07-22-multi-stage-manifest-and-post-diarize-design.md](./2026-07-22-multi-stage-manifest-and-post-diarize-design.md).
> Part 2 (`--enroll-raw`) still applies.

Decouple speaker diarization from summarization, and make it cheap to seed the
voiceprint store from an already-labeled multi-speaker recording.

## Motivation

- `full` bundles `diarize + summary`. We want "transcript + speakers" as its own
  deliverable, produced separately from "summary + pretty".
- Diarization needs audio; it cannot be added later from a `--text` raw (no
  speakers, and the raw doesn't carry the waveform). So the speaker decision is
  made at transcription time, in a dedicated fresh-audio mode.
- Speaker embeddings are already computed for free during diarization and
  persisted in `RawDoc.speakers_meta[*].embedding`. A labeled multi-speaker raw
  is a ready-made source of named voiceprints.

## Part 1 — `--diarize` mode (full minus summary)

Fresh-audio mode: `normalize → ASR + diarize + merge + voiceprints → render`.
No summary, no pretty. Produces raw with speakers and `summary=None`, plus a
speaker-tagged markdown transcript.

It slots into the existing dispatch with almost no new logic:

- `cli.py`: add `--diarize` flag; `resolve_mode` returns `"diarize"`.
- `pipeline.py`: `_safe_stage_b` already branches `if mode == "text"` (ASR only)
  else diarize+merge+voiceprints — `"diarize"` takes the else path. `_safe_stage_c`
  only summarizes when `mode == "full"`, so `"diarize"` skips summary. No stage
  changes needed beyond documenting the new mode value.
- `__main__.py`: `cmd_run` routes `mode in ("full", "text", "diarize")` to
  `run_all`.

Downstream: `--summary` picks the raw up (`summary is None`), `--pretty`
completes it. Both operate on the raw, no audio needed.

Resume: same as `full`/`text` — `scan_and_hash` skips content-hashes already
`done`. `--diarize --names "Dad,Kate"` auto-enrolls confirmed names into the
voiceprint store (existing `_apply_voiceprints` behavior).

Known limitation (accepted): a file already processed as `--text` is `done` with
no speakers; `--diarize` will skip it. To diarize it, force a re-run.

## Part 2 — `--enroll-raw <path|name>`

Backfill the voiceprint store from a raw JSON whose speakers are already named
(e.g. produced by `--diarize --names`, or hand-corrected).

- `voiceprints.py`: `enroll_named_speakers(doc, store) -> list[str]` — for each
  `SpeakerMeta` with both `name` and `embedding`, `store.enroll(name, embedding)`;
  returns the names enrolled. Pure over the store.
- resolver: given a path or a substring, find raw docs under `systems/raw/*.json`
  matching by file path, filename stem, or `source_name`.
- `cli.py` / `__main__.py`: `--enroll-raw` flag + `cmd_enroll_from_raw`.

`--enroll NAME` (single recording, dominant speaker) is unchanged.

## Part 3 — issue only (no code)

File a GitHub issue: sample-based monologue pre-check to skip diarization on solo
recordings (VAD / speaker count over a few windows, like lang-detect), saving
pyannote memory in `full`/`--diarize`.

## Testing

- `test_cli`: `resolve_mode(["--diarize"]) == "diarize"`; flag parses.
- `test_pipeline`: diarize mode writes raw+md, marks done, calls diarize, does
  NOT call summarize, and the markdown has no summary section.
- `test_voiceprints`: `enroll_named_speakers` enrolls only named+embedded
  speakers and skips unnamed / embedding-less ones; returns enrolled names.
