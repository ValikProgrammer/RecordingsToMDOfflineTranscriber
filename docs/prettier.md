# Pretty transcript (`transcriber.stages.pretty`)

LLM-rewritten, readable version of a transcript — same talk, cleaned up, not
shortened.

## What it does

Rewrites the verbatim transcript into topic-based blocks (each headed by the
earliest `[MM:SS]` timecode of that topic), fixes obvious mishearings, adds
punctuation and strips filler, while keeping ~80% of the content — every
distinct subject, fact, name and number stays. Never invents content. Like the
summary prompt, it treats ASR nonsense / repeated-word noise as recognition
errors, not real speech.

The result is written as a full document (frontmatter + summary + the rewritten
transcript) to `out/pretty/<name>.md`, alongside the normal verbatim `.md`.

## When / why

`--pretty` is a modifier, not a mode: it runs as an extra step of whatever pass
is producing output. So it works both during a fresh transcription and as a
"separate pass" over existing raw docs:

```bash
# during transcription
python -m transcriber --input-folder ./audio --pretty

# separate pass over existing raw docs
python -m transcriber --rerender --pretty     # re-render md + pretty for all
python -m transcriber --summary --pretty       # summarize + pretty
```

## Incremental `--pretty`

The pretty rewrite is the expensive LLM step, so it is skipped when the
manifest `pretty` stage is already `done`. `--force` regenerates.

- `--pretty` → skip when `stages.pretty` is `done`.
- `--pretty --force` → regenerate every pretty (same `--force` as
  [`--summary`](Summary.md)).

Skipped files are logged (`pretty stage done, skipping (use --force to redo)`).

The output still lands at `out/pretty/<name>.md` (stable name from the
manifest `out_path`).

### Combining with `--summary`

Summary skip and pretty skip are orthogonal — each checks its own signal:

- `--summary --pretty` fully handles new raw docs (summarize + pretty). Note
  `--summary` is itself incremental, so raw docs that already have a summary are
  filtered out before this step and won't get a pretty here — use `--rerender
  --pretty` to backfill.
- `--rerender --pretty` re-renders every doc's `.md` (cheap) and generates only
  the missing pretties (the expensive part is skipped where it already exists).
  This is the way to give already-summarized docs a pretty.

### Flags

| Flag | Meaning |
|------|---------|
| `--pretty` | also write the readable pretty transcript; skip if it exists |
| `--force` | with `--pretty`, regenerate even if the pretty file exists |

## Model

Ollama (`llm_model` from config, default `qwen2.5:14b`), same client as
[summarize](Summary.md). Long transcripts are chunked to fit `llm_ctx` and each
chunk rewritten in turn, then joined.
