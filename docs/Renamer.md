# Renamer (`transcriber.rename`)

LLM-driven pass that fixes poor titles on already-generated `.md` docs.

## What it does

Renames output docs in `./out` whose title is an auto-generated device name
(`Dec 6, 23 57`, `New Recording 5`) into a meaningful one derived from the doc's
own summary. Renames the file, rewrites the in-doc `Title:` / `# heading`, and
does the same for the `out/pretty/` twin.

## When / why

Run it after generating docs. The main pipeline keeps a meaningful source
filename as the title but only falls back to an LLM title for names it recognises
as "technical" — device auto-names it doesn't recognise leak through as ugly
titles. This is the post-hoc cleanup for those.

- Source audio is never touched.
- The original name stays in each doc's `Source file:` frontmatter.
- The date in the filename (`YYYY-MM-DD — <title>.md`) is resolved
  **algorithmically, not by the LLM**: frontmatter `Date:` → date in the current
  filename → file mtime. The LLM only proposes the title text.
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
#    and fills new_title / new_name
python -m transcriber.rename --propose
#    review new_name / new_title

# 3. apply — rename files (collision-safe), rewrite in-doc titles + pretty twin
python -m transcriber.rename --apply
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--folder` | `./out` | folder of generated `.md` docs |
| `--plan` | `rename_plan.json` | plan file path |
| `--batch-size` | `576` | files per LLM call (Excel-labelled A, B, … AA) |
| `--model` | config `llm_model` | Ollama model |
| `--pretty-subdir` | `pretty` | subfolder of the readable twins |
| `--config` | — | explicit config path |

## Model

Ollama (`llm_model` from config, default `qwen2.5:14b`), same client as the
summarize stage. Classify sees only filenames; propose additionally reads each
doc's summary + topics — so the expensive context is sent only for the subset
that actually needs renaming.
