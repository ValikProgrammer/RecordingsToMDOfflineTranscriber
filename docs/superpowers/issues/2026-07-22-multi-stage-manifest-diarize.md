## Summary

Manifest tracks per-stage status (`text` / `diarize` / `summary` / `pretty`). `--diarize` becomes a post-pass on existing raws (no re-ASR). Sample-based mono pre-check skips full-file pyannote on solo recordings (closes #35).

## Spec

`docs/superpowers/specs/2026-07-22-multi-stage-manifest-and-post-diarize-design.md`

## Why

~30 already-transcribed files need speakers. Today root `status=done` blocks that, and there is no per-stage visibility. Summary/pretty already support later passes but gate on raw/file existence instead of the manifest.

## Acceptance

- [ ] Legacy `done` migrates to `text=done` only; other stages `pending`
- [ ] `--diarize` on text-only raws adds speakers without re-ASR
- [ ] No raw / no text → `diarize=skipped` reason `no_transcript`
- [ ] Mono pre-check → `diarize=skipped` reason `mono` (no full pyannote)
- [ ] Pre-check failure → fallback to full diarize
- [ ] `--summary` / `--pretty` gate on manifest stages
- [ ] `--text --diarize` runs ASR∥diarize together
- [ ] Docs updated (`Diarization.md`, Summary, prettier)

## Closes

#35
