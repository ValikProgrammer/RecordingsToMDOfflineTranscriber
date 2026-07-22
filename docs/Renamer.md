# Renamer (`transcriber.rename`)

LLM-driven pass that fixes poor titles on already-generated `.md` docs **and
renames the source audio to match**.

## What it does

Renames recordings whose title is an auto-generated device name (`Dec 6, 23 57`,
`New Recording 5`) into a meaningful one derived from the doc's own summary. On
apply it renames, per recording:

- the source audio file (in `--audio-folder`, default config `input_folder`);
- the `.md` doc and its `out/pretty/` twin;

and rewrites the in-doc `Title:` / `# heading` / `Source file:` to match, plus
syncs the `systems/manifest.json` entry.

## When / why

Run it after generating docs. The main pipeline keeps a meaningful source
filename as the title but only falls back to an LLM title for names it recognises
as "technical" — device auto-names it doesn't recognise leak through as ugly
titles. This is the post-hoc cleanup for those, and it keeps the recording file
named the same as its note.

- The filename stays `YYYY-MM-DD — <title>.<ext>` (audio gets the same base as
  the `.md`, with its own extension).
- The date is resolved **algorithmically, not by the LLM**: frontmatter `Date:`
  (from Obsidian) → date in the current filename → file mtime. The LLM only
  proposes the title text.
- Audio is matched to a doc via that doc's `Source file:` frontmatter. If the
  audio is missing, the `.md` is still renamed and its `Source file:` is left
  as-is.
- It does **not** update Obsidian `[[old name]]` backlinks — run it before
  cross-linking freshly generated notes.

## How to run

Three stages sharing one editable `rename_plan.json`. Review/edit the plan
between stages; nothing is renamed until `--apply`.

```bash
# 1. classify — LLM sees ONLY filenames, flags which to rename (cheap, all files)
python -m transcriber.rename --classify --folder ./out
#    review rename_plan.json: flip "action" between "rename" / "keep"

# 2. propose — for the flagged subset only, LLM reads name + summary + topics
#    and fills new_title / new_name / new_audio_name
python -m transcriber.rename --propose
#    review new_name / new_title / new_audio_name

# 3. apply — rename audio + .md (collision-safe) + pretty twin, rewrite in-doc
#    Title / heading / Source file, sync manifest
python -m transcriber.rename --apply
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--folder` | `./out` | folder of generated `.md` docs |
| `--plan` | `rename_plan.json` | plan file path |
| `--audio-folder` | config `input_folder` | where the source audio lives |
| `--no-manifest` | off | skip the `systems/manifest.json` sync |
| `--batch-size` | `576` | files per LLM call (Excel-labelled A, B, … AA) |
| `--model` | config `llm_model` | Ollama model |
| `--pretty-subdir` | `pretty` | subfolder of the readable twins |
| `--config` | — | explicit config path |

## Model

Ollama (`llm_model` from config, default `qwen2.5:14b`), same client as the
summarize stage. Classify sees only filenames; propose additionally reads each
doc's summary + topics — so the expensive context is sent only for the subset
that actually needs renaming.
