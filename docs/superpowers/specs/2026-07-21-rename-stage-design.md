# Rename stage — design spec

Date: 2026-07-21

## Problem

Output `.md` files are named `YYYY-MM-DD — <title>.md` at generation time
(`transcriber/pipeline.py:76 resolve_title_and_date`). If the source filename
looks "technical" (regexes in `naming.is_technical_name`) the LLM-generated title
is used; otherwise the source filename is kept as the title. Auto-generated
voice-memo names (iPhone `Dec 6, 23 57`, `New Recording N`) are not matched by
those regexes, so they leak through as ugly titles
(e.g. `out/2026-07-20 — Dec 6, 23 57.md`).

Rather than widen the regexes or add a template system, we add a separate,
manually-run **`rename` post-pass** over already-generated docs. All decisions are
made by the LLM — no templates. Generation logic (`naming.py`, `pipeline.py`) and
source audio are not touched. The original filename is already preserved in the
doc frontmatter (`Source file:`) and stays.

## Three stages

Split so the expensive context (summary+topics) is only sent for the subset that
actually needs renaming (often a minority of files). Single evolving
`rename_plan.json`; the user may hand-edit it after any stage.

### 1. `--classify` — `python -m transcriber.rename --classify --folder ./out`
- Scan `out/*.md` (top level, not `pretty/`).
- Batch into groups of ≤ `--batch-size` (default 576). Within a batch, assign
  Excel-style column labels (A..Z, AA, AB, …). The LLM sees **only filenames**.
- LLM returns JSON `{"rename": ["A","C",...]}` — labels whose name is poor /
  auto-generated and worth rewriting. Everything else = keep.
- Write `rename_plan.json`:
  `{"folder","files":[{"file","action":"rename"|"keep","reason"}]}`.

### 2. `--propose` — `python -m transcriber.rename --propose`
- Read plan, take only `action == "rename"`.
- For each, parse `### Summary` text and `**Topics:**` terms from the `.md`.
- Batch (Excel labels) → LLM sees `name + summary + topics` → returns
  `{"A": "<new title>", ...}`.
- New filename via `naming.build_output_filename(day, new_title)`. `day` is
  resolved algorithmically (never from the LLM): frontmatter `Date:` (from
  Obsidian) → date in the current filename → file mtime. Filename stays
  `YYYY-MM-DD — <title>.md`.
- Fill `new_title` and `new_name` into the plan.

### 3. `--apply` — `python -m transcriber.rename --apply`
For each `rename` entry with `new_name`/`new_title`:
- Rewrite frontmatter `Title:` (via `render.yaml_escape`) and the first `# …`
  heading to `new_title`.
- Rename the file via `naming.resolve_collision(folder, new_name)` (never clobbers
  a different existing doc).
- If `out/pretty/<old name>` exists: rename it to the same `new_name` and rewrite
  its `# …` heading (pretty docs have no frontmatter).
- Log `old -> new`.

## Reuse (unchanged)

- `transcriber/stages/summarize.py:94 call_ollama_json`
- `transcriber/naming.py`: `extract_date_from_name`, `extract_date_from_file`,
  `build_output_filename`, `sanitize_filename_component`, `resolve_collision`
- `transcriber/stages/render.py:15 yaml_escape`
- `transcriber/config.py Config.llm_model` (overridable via `--model`)

Module `transcriber/rename.py` mirrors `transcriber/trim.py` structure.

## Non-goals

- No templates — LLM only.
- Do not touch generation (`naming.py`/`pipeline.py`).
- Do not fix Obsidian `[[old name]]` backlinks (fresh notes, out of scope; noted
  as a known limitation in README).
- Excel labels live only in prompts; `rename_plan.json` uses real filenames.

## Tests (`tests/test_rename.py`)

Fake `ollama` module via monkeypatch (as in `tests/stages/test_summarize.py`).
- `excel_label`: 0→A, 25→Z, 26→AA, 27→AB.
- `.md` parse: Summary/Topics extraction; no-Topics case; date from name.
- classify: label↔name mapping; plan written with actions.
- propose: `new_name`/`new_title` filled; sanitization; date from name.
- apply: rename + `Title:`/`# …` rewrite; pretty twin renamed & heading rewritten;
  keep and collision cases.

## Verification

1. `pytest tests/test_rename.py -q`.
2. Manual on `out/` (Ollama running): `--classify` → review `rename_plan.json`
   (`Dec 6, 23 57` → rename, `Natalia talk about scheduler` → keep) → `--propose`
   → review → `--apply` → confirm `.md` + `pretty/` twin renamed and titles match.
3. `pytest -q`.
