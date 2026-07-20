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
the models (`--warmup`). Don't forget to accept the model terms at
https://huggingface.co/pyannote/speaker-diarization-3.1 — otherwise
diarization fails with a clear error on first run.

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

# requeue previously failed files
python -m transcriber --retry-failed
```

### Flags

| Flag | Action |
|---|---|
| `--input-folder` / `--folder` / `--input` | audio folder |
| `--out` | output folder (default `./out`) |
| `--config` | path to `config.toml` |
| `--only NAME` | process a single file |
| `--skip NAME...` / `--exclude` | skip the given files |
| `--transcribe` / `--text` | transcript only |
| `--summary` | summary only, from existing raw JSON |
| `--resummarize` | regenerate summary from raw JSON |
| `--rerender` | rebuild `.md` from raw JSON without ASR/LLM |
| `--retry-failed` | requeue `failed` files |
| `--turbo` | use `large-v3-turbo` |
| `--speakers` / `--min-speakers` / `--max-speakers` | diarization hints |
| `--names "Me,Kate"` | names in order of first appearance in the audio |
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

## Layout

```
out/                    # .md only — this is a subfolder of the Obsidian vault
systems/raw/<hash>.json # raw transcript+diarization — source of truth
systems/manifest.json   # processing status, dedup by content hash
systems/voiceprints/    # groundwork for future voice-ID
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
- **Voice-ID is groundwork, not an implementation.** `stages/diarize.py`
  already extracts a per-speaker embedding into `speakers_meta`;
  `voiceprints.py` provides the interface (`enroll`/`identify`), but the
  methods are no-ops. When voice-ID is needed, the embeddings will already be
  in every raw JSON — no ASR/diarize rework required.

## Obsidian

Point `--out` at a subfolder inside an Obsidian vault — only `.md` files land
there. `systems/` and `logs/` sit next to `out/`, outside the vault (the
default; if you do put them inside the vault, exclude them via
`.obsidianignore`/Excluded files). Notes come with YAML frontmatter
(`title`/`date`/`language`/`speakers`/`duration`/`source_file`/`tags`) —
ready for Properties, the tag pane, Dataview, and the graph.
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
