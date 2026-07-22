# Summary (`transcriber.stages.summarize`)

Ollama-backed pass that generates a title, summary text, topics, hashtags,
key topics and decisions for a transcript, stored on the raw doc and rendered
into the output `.md`.

## What it does

For each transcript it asks the LLM for one JSON object:

- **title** — used only when the source filename is a technical/device name
  (otherwise the filename wins; see [Renamer](Renamer.md) for post-hoc fixes).
- **summary** — neutral, impersonal prose. Length is guided by a duration tier
  (below), not padded to a fixed count.
- **topics** — one entry per distinct subject, chronological, not capped.
- **hashtags**, **key_topics**, **decisions** — the last two only for long-form
  recordings (see `long_form_from_min`); short recordings get empty lists.

The summary is written back into the raw JSON (`raw/<hash>.json`, the `summary`
field) and rendered into the doc. Voice adapts to mono vs dialogue, and the
prompt tells the model to treat ASR nonsense / repeated-word noise as
recognition errors rather than real speech.

## When / why

Runs automatically in `full` mode. It is skipped in `--transcribe` (text-only)
mode — those raw docs get no summary until you run a summary pass over them.

## How to run

```bash
# full pipeline — summarize as part of transcription
python -m transcriber --input-folder ./audio

# summarize existing raw docs (e.g. after --transcribe): INCREMENTAL —
# only raw docs that don't have a summary yet
python -m transcriber --summary

# re-summarize everything, ignoring existing summaries
python -m transcriber --summary --force

# re-summarize all (legacy alias of --summary --force)
python -m transcriber --resummarize
```

### Incremental `--summary`

`--summary` mirrors the transcriber's skip-done behaviour: it summarizes only
raw docs whose `summary` is still empty. An interrupted `--summary` run resumes
without redoing finished files. The "already done" signal is the `summary`
field inside each `raw/<hash>.json` — no extra manifest state.

- `--summary` → skip raw docs that already have a summary.
- `--summary --force` → summarize every raw doc.
- `--resummarize` → re-summarize all (unchanged; equivalent to `--summary
  --force`).
- `--rerender` → re-render the `.md` only, never summarizes.

Skipped files are logged (`skipping N already-summarized (use --force to
redo)`), not silently dropped.

### Flags

| Flag | Meaning |
|------|---------|
| `--summary` | summarize existing raw docs without a summary (incremental) |
| `--force` | with `--summary`, re-summarize even already-summarized docs |
| `--resummarize` | re-summarize all raw docs (alias of `--summary --force`) |
| `--rerender` | re-render `.md` from raw, no summarization |
| `--llm-model` | override the Ollama model |
| `--only` | restrict to raw docs whose filename stem matches |

## Length tiers

Summary length scales with recording duration (config `summary.tiers`,
`sentences` is a guideline, not a hard cap):

| Duration | Sentences |
|----------|-----------|
| ≤ 15 min | 3–5 |
| ≤ 45 min | 5–8 |
| ≤ 90 min | 8–12 |
| longer | 10–15 |

Recordings at or above `long_form_from_min` (default 45) also get `key_topics`
and `decisions`.

## Model

Ollama (`llm_model` from config, default `qwen2.5:14b`). Long transcripts are
chunked to fit `llm_ctx` and summarized map-reduce: each chunk summarized, then
a reduce call merges the partials while keeping all distinct topics.
