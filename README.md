# transcriber

Fully offline pipeline: a folder of audio recordings → one Markdown note per
recording (title, timestamped summary, hashtags, transcript with roles and
timestamps). Data never leaves the machine. Target platform is a MacBook Pro
M-series (Metal/MPS).

## Setup

```bash
./setup_mac.sh
```

The script is idempotent: it installs `ffmpeg` (Homebrew), creates a `.venv`
and installs dependencies, checks for `ollama` and pulls `qwen2.5:14b`, asks
once for a HuggingFace token for `pyannote` (saved to `.env`), and warms up
the models (`--warmup`). Accept the model terms for **both** gated repos —
https://huggingface.co/pyannote/speaker-diarization-3.1 and
https://huggingface.co/pyannote/speaker-diarization-community-1 (pyannote 4.x
pulls the latter) — otherwise diarization fails with a 403 on first run.

## Usage

```bash
# full run: transcript + summary
python -m transcriber --input-folder ./audio

# what would be done, without processing
python -m transcriber --dry-run --input-folder ./audio

# transcript only, no roles, no LLM
python -m transcriber --text --input-folder ./audio

# summary only, from existing systems/raw/*.json (Whisper is not run)
python -m transcriber --summary

# regenerate summary/title/hashtags from raw JSON
python -m transcriber --resummarize

# rebuild .md from raw JSON without ASR or LLM
python -m transcriber --rerender

# one file, fast mode, known speaker count, names by speaking order
python -m transcriber --only "call with Kate" --turbo --speakers 2 --names "Me,Kate"

# force a non-default language ("" / "auto" = auto-detect)
python -m transcriber --language en

# also write a cleaned-up, readable version to out/pretty/
python -m transcriber --pretty

# enroll a voice from clean sample audio so it's auto-named in future runs
python -m transcriber --enroll "Kate" --input-folder ./sample

# requeue previously failed files
python -m transcriber --retry-failed
```

### Flags

| Flag | Action |
|---|---|
| `--input-folder` / `--folder` / `--input` | audio folder |
| `--out` | output folder (default `./out`) |
| `--systems-folder` | state folder (manifest/raw), default `./systems` |
| `--logs-folder` | logs folder, default `./logs` |
| `--config` | path to `config.toml` |
| `--backend mlx\|faster-whisper` | ASR backend (default `mlx`, Metal/GPU) |
| `--beam N` | beam size for `faster-whisper` (default 5) |
| `--only NAME` | process a single file |
| `--skip NAME...` / `--exclude` | skip the given files |
| `--transcribe` / `--text` | transcript only |
| `--summary` | summary only, from existing raw JSON |
| `--resummarize` | regenerate summary from raw JSON |
| `--rerender` | rebuild `.md` from raw JSON without ASR/LLM |
| `--retry-failed` | requeue `failed` files |
| `--turbo` | use `large-v3-turbo` |
| `--language CODE` | force decode language (default from config, `ru`); `""`/`auto` = auto-detect |
| `--speakers` / `--min-speakers` / `--max-speakers` | diarization hints |
| `--names "Me,Kate"` | names in order of first appearance; also enrolls their voiceprints |
| `--enroll NAME` | enroll a voice from the input folder's audio into the voice DB, then exit |
| `--pretty` | also write a full readable document (frontmatter + summary + an LLM-cleaned, topic-blocked transcript) to `out/pretty/` |
| `--no-frontmatter` | skip the Obsidian YAML frontmatter |
| `--wikilink-speakers` | render named speakers as `[[Name]]` |
| `--llm-model NAME` | Ollama model |
| `--jobs N` | workers for CPU stages (ffmpeg/summary/render) |
| `--diarize-device mps\|cpu` | device for pyannote |
| `--dry-run` | show the plan without processing |
| `--warmup` | download/warm up models and exit |
| `-v` / `--verbose` | verbose output |

Config is looked up at `./config.toml`, then
`~/.config/transcriber/config.toml`; template at `config.example.toml`. CLI
flags override the config.

Notable config keys (see `config.example.toml` for all): `asr_language`
(set `"auto"` to detect the language from sampled 30s windows at start/middle/end
instead of trusting Whisper's first-30s guess — needs `faster-whisper`; a concrete
value or `--language` skips detection), `lang_detect_min_prob` (per-window confidence
floor so noise can't pick a weird language), `asr_backend` (`mlx` | `faster-whisper`),
`asr_prompt_extra` (inline glossary
terms/names to bias ASR), `asr_prompt_file` (path to a git-ignored glossary file,
one term per line — keep personal names out of the repo), `asr_artifact_denylist_extra`
(extra hallucination phrases to strip), `min_speaker_share` (fold phantom
low-speech speakers; `0` = off), `voiceprint_enabled` / `voiceprint_threshold`
(auto-name speakers from the voice DB).

## Comparing ASR backends (CPU vs GPU)

Two ASR backends are available: `mlx` (default, Metal/GPU, greedy decode) and
`faster-whisper` (CTranslate2, CPU, supports beam search). To A/B the transcript
quality on the same files, run each into its **own** `--out`/`--systems-folder`/
`--logs-folder` so their manifests and outputs never collide, then diff the results.
Both use `--text` (transcript only — no diarization, no LLM), and since they use
different compute units (Metal vs CPU) they can run **at the same time**:

```bash
pip install faster-whisper   # one-time; also downloads the CT2 large-v3 model on first run

# GPU (mlx, default) — Metal
python -m transcriber --text --backend mlx \
  --input-folder ./cmp/audio --out ./cmp/gpu/out \
  --systems-folder ./cmp/gpu/systems --logs-folder ./cmp/gpu/logs

# CPU (faster-whisper, beam search) — run in a second terminal, in parallel
python -m transcriber --text --backend faster-whisper --beam 5 \
  --input-folder ./cmp/audio --out ./cmp/cpu/out \
  --systems-folder ./cmp/cpu/systems --logs-folder ./cmp/cpu/logs

# then compare
diff -u ./cmp/gpu/out/*.md ./cmp/cpu/out/*.md
```

`faster-whisper` on a Mac runs on CPU (no Metal backend in CTranslate2), so it is
several times slower than `mlx` — fine for an unattended/overnight run. The model
is still `large-v3`; beam search mainly refines word choices on ambiguous audio and
tightens timing, so proper-noun accuracy still comes from the glossary, not the backend.

## Trimming silence (preprocessing)

`transcriber.trim` is a separate, **algorithmic** step (ffmpeg `silencedetect`,
no LLM) for long recordings that are mostly silence. It never overwrites
originals — trimmed copies go to a new folder, and you confirm the cuts before
anything is written.

```bash
# 1. detect: scan the folder, write a reviewable plan
python -m transcriber.trim --input-folder ./audio

# 2. review/edit trim_plan.json: drop pairs, adjust timecodes, set "action": "skip"

# 3. apply: cut the confirmed ranges into a new folder (originals untouched)
python -m transcriber.trim --apply --out ./out/edited

# then run the normal pipeline on the trimmed copies
python -m transcriber --input-folder ./out/edited --out ./out --pretty
```

`trim_plan.json` lists, per file, `duration_sec`, `total_cut_sec`, an `action`
(`trim`/`skip`), and `cuts` — `[start, end]` pairs (seconds) marking the
regions to **remove**. Files whose total cut is below `--min-total` are marked
`skip`. Tuning flags: `--noise-db` (silence loudness threshold, default `-30`),
`--min-gap` (shortest silence worth cutting, default `20`s), `--min-total`
(skip files with less total silence than this, default `60`s), `--plan` (plan
path, default `trim_plan.json`).

Note: cutting interior silence shifts the timeline, so transcript timecodes for
edited files refer to the trimmed audio, not the original recording.

## Renaming poor titles (post-processing)

`transcriber.rename` is a separate, **LLM-driven** pass over already-generated
`.md` docs. The pipeline keeps a meaningful source filename as the title but
falls back to an LLM title only for names it recognises as "technical"; device
auto-names it doesn't recognise (`Dec 6, 23 57`, `New Recording 5`) leak through
as ugly titles. This pass fixes them after the fact — and renames the **source
audio** to match, so the recording and its note stay in sync.

Three stages so the model only reads summaries for the subset that needs it:

```bash
# 1. classify — LLM sees ONLY filenames and flags which to rename (cheap)
python -m transcriber.rename --classify --folder ./out
# review/edit rename_plan.json: flip "action" between "rename"/"keep"

# 2. propose — for the flagged subset, LLM reads name + summary + topics and
#    proposes new_title / new_name / new_audio_name
python -m transcriber.rename --propose
# review/edit new_name / new_title / new_audio_name

# 3. apply — renames the source audio, the .md (collision-safe) and its
#    out/pretty/ twin, and rewrites the in-doc Title / heading / Source file;
#    also syncs systems/manifest.json
python -m transcriber.rename --apply
```

The filename stays `YYYY-MM-DD — <title>.<ext>` (audio gets the same base as the
`.md`, its own extension). The LLM only proposes the title text, never the date:
the date is resolved algorithmically — frontmatter `Date:` (from Obsidian) →
date in the current filename → file mtime. The source audio is located via each
doc's `Source file:` frontmatter, in `--audio-folder` (default config
`input_folder`). If the audio is missing, the `.md` is still renamed and its
`Source file:` is left as-is.

Flags: `--folder` (docs folder, default `./out`), `--plan` (default
`rename_plan.json`), `--audio-folder` (default config `input_folder`),
`--no-manifest` (skip the manifest sync), `--batch-size` (files per LLM call,
default `576`), `--model` (defaults to config `llm_model`), `--pretty-subdir`
(default `pretty`).

Known limitation: it does not update Obsidian `[[old name]]` backlinks, so run it
before cross-linking freshly generated notes.

## Layout

```
out/                    # .md only — this is a subfolder of the Obsidian vault
out/pretty/             # full readable documents: frontmatter + summary + cleaned transcript (only with --pretty)
out/edited/             # silence-trimmed audio copies (transcriber.trim --apply)
trim_plan.json          # reviewable silence-cut plan (transcriber.trim)
rename_plan.json         # reviewable title-rename plan (transcriber.rename)
systems/raw/<hash>.json # raw transcript+diarization — source of truth
systems/manifest.json   # processing status, dedup by content hash
systems/voiceprints/    # enrolled voice embeddings, one JSON per name
logs/run.log            # run log
logs/<name>__<hash8>.log # per-file log
```

## Architecture and trade-offs

- **Single ASR backend (`mlx-whisper`) for every file.** MLX gives a native
  Metal path on Apple Silicon; whisperX/CTranslate2 runs on CPU on Mac and
  would be slower without a quality gain on roles — roles are assigned by a
  separate layer (`stages/merge.py`) on top of Whisper's word-level
  timestamps, so a second ASR backend isn't needed.
- **`systems/raw/<hash>.json` is the source of truth.** Whisper and pyannote
  run once; summary/title/render are pure functions of that JSON, so
  `--resummarize`/`--rerender` are instant and never touch the GPU.
- **Staged pipeline, not full parallelism.** ffmpeg and summary/render scale
  with `--jobs` workers (CPU-bound), but ASR+diarize run on a **single**
  dedicated thread: Metal/MPS doesn't share context well across parallel
  Whisper runs, so parallelizing transcription itself would just risk memory
  pressure for no gain. While the GPU thread processes file N, ffmpeg is
  preparing N+1 and Ollama is finishing up N−1 — a pipeline, not a barrier.
- **Dedup by content hash (blake2b), not by name.** Renaming a file doesn't
  trigger reprocessing; changing its content does. The manifest writes
  `in_progress` **before** the heavy work and `done` **only after** raw+md
  are written atomically, so an interruption never leaves a "half" file
  marked done.
- **Voice-ID by embedding, not a fixed model.** `stages/diarize.py` extracts a
  per-speaker embedding into `speakers_meta`; `voiceprints.py` enrolls those
  under names (via `--names` or `--enroll`) and matches new speakers by cosine
  similarity, so `SPEAKER_00` becomes e.g. `Kate` across recordings. Confirmed
  `--names` speakers are the ground truth that gets enrolled; auto-matches only
  relabel, they don't feed back into the store.

## Obsidian

Point `--out` at a subfolder inside an Obsidian vault — only `.md` files land
there. `systems/` and `logs/` sit next to `out/`, outside the vault (the
default; if you do put them inside the vault, exclude them via
`.obsidianignore`/Excluded files). Notes come with YAML frontmatter
(`Title`/`Date`/`Language`/`Speakers`/`Duration`/`Source file`, plus lowercase
`tags` so Obsidian treats them as real tags) — ready for Properties, the tag
pane, Dataview, and the graph.
`--wikilink-speakers` renders named speakers as `[[Name]]` for backlinks and
graph nodes; `Speaker 1/2` is never turned into a wikilink. `--no-frontmatter`
disables the frontmatter if you don't need it.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

ASR/diarize/LLM are mocked (the `mlx_whisper`/`pyannote.audio`/`ollama`
modules are imported lazily inside their functions; tests substitute fake
modules via `sys.modules`); `ffmpeg` normalization and all pure logic
(naming, merge, manifest, render, pipeline orchestration) are tested
directly.
