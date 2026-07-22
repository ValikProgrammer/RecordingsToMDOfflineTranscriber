# Resumable `--summary` — design

## Problem

`--summary` / `--resummarize` / `--rerender` go through `run_existing`, which
reprocesses **every** raw JSON. Unlike the audio pipeline (`scan_and_hash`
diffs against the manifest so `done` files become `skip`), there is no
"summary already done → skip". An interrupted `--summary` run redoes
everything from scratch.

## Goal

Make `--summary` incremental, mirroring the transcriber's skip-done behavior:
summarize only raw docs that don't yet have a summary. Keep a way to force a
full re-run.

## Behavior

- `--summary` → incremental. Summarize only raw docs where `summary` is absent.
- `--summary --force` → summarize every raw doc (ignore existing summaries).
- `--resummarize` → unchanged (re-summarize all). Effectively an alias of
  `--summary --force`; kept for back-compat and existing tests.
- `--rerender` → unchanged (never summarizes).

## "Already done" signal

The raw JSON carries `summary` inline. After `--text` it is `None`; after
`full` / `--summary` it is populated and written back to the raw file. So the
signal lives in the file itself — no new manifest field.

Rule: `doc.summary is None` → needs summary. A populated `summary` → skip.

## Changes

- `cli.py`: add `--force` flag (`dest="force"`, `action="store_true"`); thread
  into `RunOptions`.
- `pipeline.py`:
  - add `RunOptions.force: bool = False`.
  - add pure helper `filter_unsummarized(raw_paths) -> list[Path]` that loads
    each doc and keeps those with `summary is None`. Testable with fake raw
    JSON, no real deps.
- `__main__.py` (`cmd_run`, else branch): when `mode == "summary" and not
  opts.force`, run `raw_paths` through `filter_unsummarized` and log how many
  were skipped (never silently drop). `resummarize` / `rerender` unchanged.

Double-load (filter + process) is accepted — matches the existing per-file
`_raw_duration` load; raw JSON is cheap.

## Tests

- `filter_unsummarized`: keeps docs without summary, drops docs with one.
- CLI: `--force` parses; `RunOptions.force` set from it.
- Existing `resummarize` pipeline tests stay green.

## Out of scope

- Prompt edits in `pretty.py` / `summarize.py` — committed separately.
- Git topology / branch-base cleanup.
- `--rerender` skip logic.
