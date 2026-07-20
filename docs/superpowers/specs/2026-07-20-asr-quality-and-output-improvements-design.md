# ASR Quality & Output Improvements — Design

Date: 2026-07-20

## Problem

On a real Russian recording (`Natalia talk about scheduler.m4a`, 22:42) the pipeline
produced poor output:

1. The quiet intro made Whisper auto-detect the language as `nn` (Norwegian
   Nynorsk). It then decoded Russian speech through the wrong language, producing
   garbage plus ~8 minutes of the classic hallucination `Thank you for watching!`
   / `Takk for watching!`.
2. `condition_on_previous_text=True` (Whisper default) made the hallucinated line
   repeat for minutes once it appeared.
3. Domain terms/names were mangled (ФизТех → "FDI", хакатон → "хакатум") because
   no vocabulary hint was given.

Two more requests came out of reviewing that output:

4. Speed: ~6–7× realtime end-to-end (ASR dominates). User wants faster for a
   ~60h backlog, but **not** at the cost of quality.
5. Topics rendering: per-record `Topics` should not be capped, each topic on its
   own line, timecode first.

Also clarified as **not bugs**: `long_form_from_min` works correctly — it gates
the `### Key Topics` / `### Decisions` sections (≥45 min); the inline `Topics`
line is intentionally always shown. No cross-file topic index is wanted.

## Root cause

`transcriber/stages/asr_mlx.py` calls `mlx_whisper.transcribe(wav,
path_or_hf_repo=repo, word_timestamps=True)` with no language, no
`initial_prompt`, and default `condition_on_previous_text`/no
`hallucination_silence_threshold`. `mlx_whisper.transcribe` accepts all of these
(verified against the installed version).

## Scope

Change only: `asr_mlx.py`, `config.py` (+ `config.example.toml`), `cli.py`,
`summarize.py`, `render.py`, `pipeline.py`, and their tests. No other stages.

## Design

### A. ASR quality (`asr_mlx.py`, `config.py`, `cli.py`)

- New config fields:
  - `asr_language: str = "ru"` — forced decode language. Empty string or
    `"auto"` means auto-detect (for non-Russian recordings).
  - `asr_prompt_extra: str = ""` — user's personal terms/names, appended to the
    built-in prompt.
- New CLI flag `--language` overriding `asr_language` for a run.
- A built-in generic Russian/IT `initial_prompt` constant seeding common
  vocabulary and correct casing, e.g.:
  `"Совещание. Обсуждаем ФизТех, хакатон, стипендию, ментора, практику, дедлайн, проект, репозиторий, деплой, бэкенд, API, Телеграм-бота, субботнюю школу."`
  The effective prompt is this constant plus `asr_prompt_extra` (space-joined,
  trimmed). Kept short (Whisper only uses ~224 prompt tokens).
- `transcribe()` gains `language: str | None` and `initial_prompt: str | None`
  parameters (defaults keep the signature backward-compatible). It passes to
  `mlx_whisper.transcribe`:
  - `language=language` (omitted when auto-detect)
  - `initial_prompt=initial_prompt`
  - `condition_on_previous_text=False`
  - `hallucination_silence_threshold=2.0`
  - existing `word_timestamps=True`
- `pipeline.py` passes `cfg.asr_language` (normalized: `""`/`"auto"` → `None`)
  and the composed prompt into `transcribe()`.

### B. Safe speedup (`pipeline.py`) — best-effort, measured

- Within stage B, run diarization (on CPU) concurrently with ASR (on Metal/GPU)
  for the same file, then merge. Expected ~1.2–1.5×, quality-neutral (same
  models, same params).
- This is explicitly best-effort: after implementation, measure on a real file.
  If concurrent diarization-on-CPU does **not** beat the current sequential
  ASR→diarize (Metal), revert to sequential and record the result. Do not
  degrade quality or correctness for speed.
- `--turbo` (large-v3-turbo) stays available for when speed is critical; it is
  **not** made the default (turbo trades a little accuracy on rare terms, which
  is exactly the pain point here).
- 3× without turbo is not promised; user accepted this ("quality first").

### C. Topics (`render.py`, `summarize.py`)

- Render each topic on its own line, timecode first:
  ```
  **Topics:**
  - [08:30] Организация хакатона
  - [12:10] Субботняя школа и расписание
  ```
  (replaces the single `" · "`-joined `term (ts)` line)
- No cap on the number of topics (there is none in code today; keep it that way).
- Summary prompt: instruct the model to extract topics proportional to content
  (long recording → many distinct topics, without padding). In the map-reduce
  `reduce` step, dedupe only exact/near-duplicate topics — do not collapse
  distinct topics — so long recordings keep all their topics.

## Testing

- Unit (`asr_mlx`): a fake `mlx_whisper` records kwargs; assert `language`,
  `condition_on_previous_text=False`, `hallucination_silence_threshold` set, and
  `initial_prompt` contains both the built-in glossary and `asr_prompt_extra`.
  Assert auto-detect path omits `language` when config is `""`/`"auto"`.
- Unit (`config`/`cli`): `asr_language`/`asr_prompt_extra` load from TOML;
  `--language` overrides.
- Unit (`render`): topics render one-per-line, timecode first, all topics
  present (e.g. 30 topics → 30 lines).
- Unit (`pipeline`): stage B still produces a correct merged result whether
  diarization runs concurrently or sequentially.
- Acceptance: re-run `Natalia talk about scheduler.m4a` (`--retry-failed`) →
  language `ru`, no `Thank you for watching` block, `ФизТех` recognized, topics
  as a timecode-first list.

## Out of scope

- Cross-file topic index (explicitly not wanted).
- Post-ASR LLM term-correction pass (possible future step if glossary is
  insufficient).
- Russian fine-tuned ASR model (MLX availability is thin; revisit only if needed).
