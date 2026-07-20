# CREATE_SYSTEM.md — Implementation brief: local offline transcription + summary pipeline

> **For:** the implementing model (Claude Sonnet 5).
> **Working style:** follow this document strictly, step by step (see §16 "Implementation order"). Don't second-guess the architecture — all decisions are already made and locked in below. If something isn't described — pick the **simplest** option and leave a `# TODO:` explaining why. Write idiomatic Python 3.11+, typed (dataclasses + type hints), with docstrings.

---

## 1. What we're building

A CLI tool `transcriber` that takes a **folder of audio recordings** and produces **one Markdown file per recording**: title, timestamped summary, hashtags, full transcript with roles and timestamps.

**Hard constraints:**
- **Fully offline.** No cloud APIs. All models are downloaded once and then run locally. Data is confidential and never leaves the machine.
- **Target machine:** MacBook Pro **M4 Max, 64 GB RAM**, macOS (Apple Silicon). Optimize for Metal/MPS.
- **Resilient to restarts:** an interruption must not corrupt data; a restart only redoes unfinished work.

---

## 2. Locked-in stack (DO NOT change)

| Component | Choice | Notes |
|---|---|---|
| Transcription (ASR) | **`mlx-whisper`**, model `mlx-community/whisper-large-v3-mlx` | Native Apple MLX (Metal). `word_timestamps=True` is required. Turbo mode: `mlx-community/whisper-large-v3-turbo`. |
| Diarization | **`pyannote.audio`**, `pyannote/speaker-diarization-3.1` | Runs on **MPS** (with a CPU fallback flag). Requires an HF token + accepting the model terms once (offline after that). |
| Word↔speaker merge | **custom code** (§9) | Assign each word to the speaker with maximum time overlap. |
| Language | **Whisper auto-detect** on every file | A separate detector is NOT needed. Expect ~90% RU, ~10% EN. |
| LLM (summary/title/hashtags) | **Ollama**, default model **`qwen2.5:14b`** | Good at Russian. `--llm-model` flag. Output strictly JSON (`format=json`). |
| Audio normalization | **`ffmpeg`** → 16 kHz mono PCM WAV | Before feeding into ASR. |
| Content hash | **blake2b**, streaming byte read | Dedup key. |

**Single ASR path:** MLX for ALL files. No `whisperX`/dual-backend. No VAD in v1 (groundwork for later — see §17).

---

## 3. Setup (`setup_mac.sh`)

One idempotent script, with clear output at every step:

```bash
#!/usr/bin/env bash
set -euo pipefail
# 1. Homebrew ffmpeg
command -v ffmpeg >/dev/null || brew install ffmpeg
# 2. Python venv
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# 3. Ollama (check install, pull the model)
command -v ollama >/dev/null || { echo "Install Ollama: https://ollama.com/download"; exit 1; }
ollama pull qwen2.5:14b
# 4. HF token for pyannote (one-time): ask the user, save to .env,
#    and remind them to accept the model terms at huggingface.co/pyannote/speaker-diarization-3.1
# 5. Warm up/download ASR and diarization models (run transcriber --warmup)
python -m transcriber --warmup
echo "Done. Check: python -m transcriber --dry-run --input-folder ./audio"
```

`requirements.txt`: `mlx-whisper`, `pyannote.audio`, `torch`, `torchaudio`, `ollama`, `tqdm`, `python-dotenv`, `tomli` (if Python <3.11). Pin versions (`==`) at implementation time.

`--warmup`: load the ASR and pyannote models (downloads on first run), run them on a 2-second silent clip, print "models ready". That way the first real run doesn't stumble over a download.

---

## 4. Project structure

```
transcriber/
  __init__.py
  __main__.py          # entry point: python -m transcriber
  cli.py               # argparse: all flags and modes (§6)
  config.py            # config.toml loading + defaults (§6.1)
  pipeline.py          # stage orchestration, queues, --jobs (§13)
  models.py            # dataclasses: Word, Segment, RawDoc, SpeakerMeta, ManifestEntry
  manifest.py          # read/atomic write of systems/manifest.json (§12)
  naming.py            # title/date/sanitization/collisions (§8)
  logging_setup.py     # shared log + per-file log (§12)
  voiceprints.py       # STUB interface for future voice-ID (§14)
  stages/
    __init__.py
    ingest.py          # folder scan + blake2b hash + "process/skip" decision
    audio.py           # ffmpeg → 16k mono wav (temp), ffprobe duration
    asr_mlx.py         # mlx-whisper transcription with word timestamps
    diarize.py         # pyannote diarization → speaker segments (+ embedding extraction)
    merge.py           # word↔speaker, monologue collapse
    summarize.py       # Ollama: title/summary/topics/hashtags (JSON)
    render.py          # RawDoc → Markdown per the template (§11)
setup_mac.sh
config.example.toml
requirements.txt
README.md
tests/                 # see §16
```

Working directories (created automatically, paths configurable):
```
<out>/                         # default: ./out  — final .md files
<out>/../systems/raw/<hash>.json     # raw result (source of truth)
<out>/../systems/manifest.json       # processed-files manifest
<out>/../systems/voiceprints/        # (groundwork) voiceprints
<out>/../logs/<name>__<hash8>.log    # per-file log (name + short hash)
<out>/../logs/run.log                # shared run log
```
> **IMPORTANT:** raw data lives in **`systems/raw/`** (not `out/.raw/`). The manifest is **`systems/manifest.json`**. The per-file log name is a human-readable name + short hash, e.g. `logs/call-with-kate__a1b2c3d4.log`.

> **Obsidian:** `out/` is a **subfolder of the Obsidian vault**. So the vault must only ever receive **`.md`** files. Keep `systems/` (raw JSON, manifest, voiceprints) and `logs/` **outside the vault** (by default they're siblings of `out/`, not inside it — keep it that way). If the user does put them inside the vault, either prefix them with `.` **or** add them to `.obsidianignore`/"Excluded files" so Obsidian doesn't index the JSON/logs. No internal files inside `out/`.

---

## 5. Data formats

### 5.1 Raw document `systems/raw/<hash>.json` — SOURCE OF TRUTH

Markdown and the summary are derived from it. Whisper runs once; the summary/format can be regenerated from this JSON instantly.

```jsonc
{
  "schema": 1,
  "content_hash": "blake2b:<hex>",
  "source_name": "call with Kate.m4a",     // original name, always kept
  "source_path": "/abs/path/call with Kate.m4a",
  "language": "ru",                         // Whisper auto-detect
  "duration_sec": 754.2,
  "num_speakers": 2,                        // after monologue collapse
  "is_monologue": false,
  "asr": { "backend": "mlx", "model": "large-v3", "turbo": false },
  "created_at": "2026-07-19T21:00:00Z",     // ISO-8601 UTC
  "segments": [
    {
      "start": 0.0, "end": 3.1,
      "speaker": "SPEAKER_00",              // or null for a monologue
      "text": "Hi, how are you…",
      "words": [ { "w": "Hi,", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00" } ]
    }
  ],
  "speakers_meta": [                         // groundwork for voice-ID (§14)
    { "label": "SPEAKER_00", "name": null, "embedding": null, "total_speech_sec": 512.3 }
  ],
  "summary": {                              // filled in by the summarize stage; absent with --text
    "title": "Call with Kate",
    "text": "…summary, length per tier/number of topics…",
    "topics": [ { "term": "war", "ts": 12.0 }, { "term": "Cyprus", "ts": 225.0 } ],
    "hashtags": ["war", "cyprus", "telegram", "relationships"],
    "key_topics": [                         // long-form only; otherwise []
      { "topic": "Discussion about moving to Cyprus", "ts": 225.0, "note": "brief gist" }
    ],
    "decisions": [                          // long-form only; otherwise []
      { "text": "Decided not to sell the apartment until autumn", "ts": 2470.0 }
    ],
    "length_tier": "5–8",                   // the applied guidance (for transparency)
    "generated": true,                      // true if the title was LLM-generated, false if from the filename
    "llm_model": "qwen2.5:14b"
  }
}
```

### 5.2 Manifest `systems/manifest.json`

Key — **content hash** (dedup by content, not by name).

```jsonc
{
  "schema": 1,
  "entries": {
    "blake2b:<hex>": {
      "content_hash": "blake2b:<hex>",
      "source_name": "call with Kate.m4a",
      "status": "done",                 // done | failed | in_progress
      "language": "ru",
      "num_speakers": 2,
      "duration_sec": 754.2,
      "out_path": "out/2026-07-12 — Call with Kate.md",
      "raw_path": "systems/raw/<hash>.json",
      "log_path": "logs/call-with-kate__a1b2c3d4.log",
      "elapsed_sec": 63.7,
      "error": null,
      "updated_at": "2026-07-19T21:01:03Z"
    }
  }
}
```

**Status rules:**
- Write `in_progress` **before** the heavy work. If the process crashes — on restart, treat `in_progress` as unfinished → reprocess.
- Write `done` **only after** a successful Markdown render + atomic save of all files.
- `failed` — with the error text; skip such files, but `--retry-failed` puts them back in the queue.

### 5.3 Per-file log format `logs/<name>__<hash8>.log`

Plain lines with a UTC timestamp. Must record: when the file was picked up, every stage (ffmpeg/ASR/diarize/merge/summary/render), **detected language**, **speaker count**, **processing time**, any errors with a stack trace.

```
2026-07-19T21:00:00Z [INFO] taken: call with Kate.m4a (hash a1b2c3d4)
2026-07-19T21:00:01Z [INFO] ffmpeg -> 16k mono wav, duration=754.2s
2026-07-19T21:00:41Z [INFO] ASR done: language=ru, segments=142
2026-07-19T21:00:55Z [INFO] diarize done: raw_speakers=2
2026-07-19T21:00:55Z [INFO] merge done: num_speakers=2, monologue=False
2026-07-19T21:01:02Z [INFO] summary done (qwen2.5:14b)
2026-07-19T21:01:03Z [INFO] rendered: out/2026-07-12 — Call with Kate.md (elapsed=63.7s)
```

---

## 6. CLI — all flags and modes

### 6.1 Config + defaults

`config.toml` (look in CWD, then `~/.config/transcriber/config.toml`; CLI flags override the config). Defaults:

```toml
input_folder = "./audio"
out_folder   = "./out"
systems_folder = "./systems"
logs_folder  = "./logs"
asr_model    = "large-v3"      # large-v3 | large-v3-turbo (see --turbo)
llm_model    = "qwen2.5:14b"
llm_ctx      = 8192            # if the transcript is longer — map-reduce the summary
diarize_device = "mps"         # mps | cpu
mono_threshold = 0.92          # dominant speaker's share of speech time above which it's treated as a monologue
jobs = 3                       # workers for CPU stages (ffmpeg/summary/render); the GPU stage is always 1
obsidian_frontmatter = true    # add a YAML frontmatter (properties) block at the top of the .md (§11)
wikilink_speakers = false      # render named speakers as [[Name]] (Obsidian graph/backlinks)

# --- Adaptive summary length (GUIDANCE, not a hard rule) ---
# The model picks the length by number of substantive topics; the range below is just a duration-based hint.
[summary]
# Use the FIRST tier where duration_min <= up_to_min; if longer than all tiers, use the last one.
tiers = [
  { up_to_min = 15,     sentences = "3–5"   },
  { up_to_min = 45,     sentences = "5–8"   },
  { up_to_min = 90,     sentences = "8–12"  },
  { up_to_min = 100000, sentences = "10–15" },   # "and longer" (a large number = no practical upper bound)
]
long_form_from_min = 45   # from this duration on, add the "Key Topics" and "Decisions" blocks (§11)
```

### 6.2 Flags

| Flag | Action |
|---|---|
| `--input-folder PATH` / `--folder` / `--input` | audio folder (all three are synonyms) |
| `--out PATH` | output folder (default `./out`) |
| `--config PATH` | path to the config |
| `--only NAME` | process a single file (by name, extension optional) |
| `--skip NAME [NAME…]` / `--exclude` | don't process the given files |
| `--transcribe` / `--text` | **transcript only** (no summary/LLM title; title from the filename or a default) |
| `--summary` | **summary only**, from existing `systems/raw/*.json` (Whisper is not run) |
| `--resummarize` | regenerate summary/title/hashtags from raw JSON (overwrites the `summary` block) |
| `--rerender` | rebuild the Markdown from raw JSON without ASR or LLM |
| `--retry-failed` | requeue files with status `failed` |
| `--turbo` | fast mode: `large-v3-turbo` model |
| `--speakers N` | fixed speaker count (passed to pyannote) |
| `--min-speakers N` / `--max-speakers N` | speaker count range |
| `--names "Me,Kate"` | speaker names in order of first appearance in the audio (§14) |
| `--no-frontmatter` | don't add YAML frontmatter to the `.md` (added by default) |
| `--wikilink-speakers` | render named speakers as `[[Name]]` (Obsidian graph) |
| `--llm-model NAME` | override the Ollama model |
| `--jobs N` | number of workers for the CPU stages |
| `--diarize-device mps\|cpu` | device for pyannote |
| `--dry-run` | show the plan (which files will be processed/skipped and why), do NOT process |
| `--warmup` | download/warm up models and exit |
| `-v/--verbose` | verbose output |

**Default modes:** with no flags — every file gets a transcript, then a summary. `--text` = transcript only. `--summary` = summary only, from existing raw. `--resummarize`/`--rerender` operate on existing raw JSON and do NOT touch Whisper.

---

## 7. Stage pipeline (signatures)

Each stage is a pure function, taking and returning dataclasses/primitives, logging to the per-file log. Order for a single file:

```python
# stages/ingest.py
def scan_and_hash(folder: Path, cfg) -> list[FileTask]:
    """Find audio (.m4a .mp3 .wav .aac .flac .ogg .mp4/.m4v/.mov -> extract audio),
    compute blake2b, check against the manifest. Return tasks with status to_do/skip/redo."""

# stages/audio.py
def normalize(src: Path, tmp_dir: Path, log) -> tuple[Path, float]:
    """ffmpeg -i src -ac 1 -ar 16000 -c:a pcm_s16le tmp.wav ; return (wav_path, duration_sec)."""

# stages/asr_mlx.py
def transcribe(wav: Path, model: str, turbo: bool, log) -> AsrResult:
    """mlx_whisper.transcribe(wav, path_or_hf_repo=..., word_timestamps=True).
    Return language + segments with words and timestamps."""

# stages/diarize.py
def diarize(wav: Path, cfg, num_speakers=None, log) -> DiarResult:
    """pyannote pipeline on MPS (CPU fallback). Return a list of (start,end,label).
    Also compute a per-speaker embedding (§14) and total_speech_sec."""

# stages/merge.py
def merge(asr: AsrResult, diar: DiarResult, mono_threshold: float, names: list[str]|None, log) -> RawDoc:
    """Assign a speaker to every word (§9); collapse a monologue; apply --names."""

# stages/summarize.py
def summarize(doc: RawDoc, cfg, log) -> Summary:
    """Ollama JSON: title/summary/topics(ts)/hashtags in the recording's language (§10)."""

# stages/render.py
def render_markdown(doc: RawDoc) -> str:
    """RawDoc -> Markdown per the template in §11."""
```

---

## 8. Naming logic (`naming.py`) — implement EXACTLY

### 8.1 Decide: "meaningful" or "technical" name

**Technical** (→ the title is LLM-generated, the date comes from the filename/metadata):
- A bare date: `2026-07-12`, `20260712`, `2026.07.12`, `12-07-2026`.
- A recorder prefix: `REC_20260712`, `AUD-20260712`, `voice 001`, `recording_003`.
- **iPhone style:** `New Recording`, `New Recording 12`, `Новая запись`, `Новая запись 5`, `Recording 3`, `Запись 7` (case-insensitive, with or without a number).
- Junk: `audio`, `voice memo`, digits/underscores only.

**Meaningful** (→ the title comes from the filename, 3–4 words) — everything else, e.g. `call with Kate`, `project sync call`.

Implement as a set of regexes (a list of "technical name" patterns; if none match — treat it as meaningful). Keep the pattern list easy to extend.

### 8.2 Date
1. Extract from the filename: regex for `YYYY-MM-DD`, `YYYYMMDD`, `DD-MM-YYYY`.
2. If none — from file metadata: creation time (`st_birthtime` on macOS), otherwise `st_mtime`.
3. Format in the output filename and Markdown metadata: `YYYY-MM-DD`.

### 8.3 Title
- Meaningful filename → take it from the filename, normalize to 3–4 significant words (strip the extension, separators → spaces, capitalize the first word).
- Technical filename → title from the LLM (summarize stage). In `--text` mode (no LLM), the title is `"Recording YYYY-MM-DD"` (or from the filename, if meaningful).

### 8.4 Output filename
- Format: `YYYY-MM-DD — <Title>.md` (date prefix for chronological sorting; ` — ` separator).
- **Sanitization:** strip/replace `/ \ : * ? " < > |` **and Obsidian-forbidden `# ^ [ ]`** and control characters; collapse whitespace; cap the title length (e.g. 60 characters). Keep the ` — ` (dash separator) — it's allowed in Obsidian filenames/links.
- **Collisions:** if the file already exists — append a ` (2)`, ` (3)`, … suffix. Check collisions against the actual path on disk.
- The original filename is ALWAYS written both to the raw JSON (`source_name`) and the Markdown metadata (`**Source file:**`).

---

## 9. Diarization, monologue, and merge (algorithm)

1. **pyannote** returns segments `(start, end, label)`. Pass `--speakers N` / `--min/--max-speakers` into the pipeline.
2. **Word↔speaker merge:** for every ASR word (which has `start/end`) assign the diarization segment's speaker with the **maximum time overlap**. If there's no overlap — use the nearest segment in time.
3. **Building Markdown segments:** group consecutive words from the same speaker into one line; a new line starts on a speaker change or a pause > ~1.5 s.
4. **Monologue collapse:**
   - Compute `total_speech_sec` per speaker.
   - If there's exactly 1 speaker → `is_monologue=True`, `speaker=null` on all segments.
   - If the dominant speaker accounts for ≥ `mono_threshold` (default 0.92) of total speech time → also treat it as a monologue (guards against a false second speaker from noise/echo).
   - Otherwise — keep all speakers.
5. **Labels:** `SPEAKER_00`, `SPEAKER_01`… Show in Markdown as `Speaker 1`, `Speaker 2` (index+1), or a name from `--names`/voice-ID (§14).

---

## 10. LLM (Ollama) — prompts and invocation

### 10.1 Invocation
`ollama.chat(model=cfg.llm_model, format="json", messages=[...])`. Always use `format="json"` — guarantees a parseable response. Retry once on bad JSON.

### 10.2 Language
Summary/topics/hashtags/title — **in the recording's language** (`doc.language`). The system prompt is bilingual (RU/EN), with the target language substituted in.

### 10.3 Input
The transcript **with timestamps** (compact, `[MM:SS] text`), so the model can stamp a timestamp on every topic. If the length exceeds `llm_ctx` — **map-reduce**: split into chunks, summarize each, then merge. Compute topics/hashtags on the final merge; take topic timestamps from the nearest segment matching the keyword.

### 10.3a Adaptive summary length (GUIDANCE, not a hard rule)

By `duration_sec`, pick a tier from the `[summary].tiers` config (§6.1): **the first tier where `duration_min <= up_to_min`; if the recording is longer than all tiers — the last one**. Substitute its `sentences` (e.g. `"5–8"`) into the prompt as **guidance**.

**Important for the prompt:** explicitly tell the model this is guidance, NOT a hard requirement — the actual length is driven by the **number of substantive topics**, not duration. Let the model deviate from the range if there's just one topic (shorter) or many (longer). Don't "pad with filler" to hit the range.

**Long-form:** if `duration_min >= long_form_from_min` (default 45) — additionally ask the model for two lists (the `key_topics` and `decisions` fields below). Otherwise these fields can be returned empty/omitted.

### 10.4 Required JSON output
```jsonc
{
  "title": "3–4 words, in the recording's language (only if the filename is technical; otherwise ignored)",
  "summary": "the summary; length ~ the tier guidance, but driven by the number of substantive topics",
  "topics": [ { "term": "war", "ts_hint": "00:12" }, { "term": "Cyprus", "ts_hint": "03:45" } ],
  "hashtags": ["war", "cyprus", "telegram", "relationships"],
  // --- Long-form only (long recordings); otherwise [] ---
  "key_topics": [
    { "topic": "Discussion about moving to Cyprus", "ts_hint": "03:45", "note": "brief gist of the topic" }
  ],
  "decisions": [
    { "text": "Decided not to sell the apartment until autumn", "ts_hint": "41:10" }
  ]
}
```
- `ts_hint` — timestamp `MM:SS` (or `HH:MM:SS`); at render time, match it to the nearest real segment.
- `hashtags` — Latin/Cyrillic, no spaces or `#` (the render adds the hash). 3–6 topics and hashtags.
- `key_topics` — the conversation's **main** topics with timestamps and a brief gist (an expanded list, unlike the compact `topics` line). `decisions` — decisions made / conclusions / agreements with a timestamp (an empty list if there are no explicit decisions).
- In long-form map-reduce: collect `key_topics`/`decisions` at the final merge from the per-chunk summaries.

---

## 11. Markdown template (`render.py`) — render it EXACTLY like this (Obsidian-formatted)

The output lands in an Obsidian vault, so it starts with **YAML frontmatter** (Obsidian properties), followed by a human-readable title and body.

```markdown
---
title: "Call with Kate"
date: 2026-07-12
language: RU
speakers: 2
duration: "12:34"
source_file: "call with Kate.m4a"
tags: [war, cyprus, telegram, relationships]
---

# Call with Kate

**Date:** 2026-07-12  ·  **Language:** RU  ·  **Speakers:** 2  ·  **Duration:** 12:34
**Source file:** `call with Kate.m4a`

### Summary
Up to 5 sentences on what was discussed.

**Topics:** war (00:12) · Cyprus (03:45) · Telegram (07:20) · relationships (09:10)

**Hashtags:** #war #cyprus #telegram #relationships

<!-- The blocks below are ONLY for long recordings (duration >= long_form_from_min) -->
### Key Topics
- **Discussion about moving to Cyprus** (03:45) — brief gist of the topic
- **The Telegram situation** (07:20) — brief gist of the topic

### Decisions
- Decided not to sell the apartment until autumn (41:10)
- Agreed to call again next week (58:03)

---

### Transcript
**[00:00] Speaker 1:** …
**[00:15] Speaker 2:** …
```

Rendering rules:
- **Frontmatter (Obsidian properties):** add by default (`obsidian_frontmatter=true`, disabled by `--no-frontmatter`). Keys as above. `date` — unquoted (Obsidian recognizes it as a date). `duration` — a quoted string. `tags` — the same hashtags, but **without `#`** and **slugified**: spaces → `-`, strip `#`, lowercase; drop purely numeric tags (Obsidian doesn't support them). Example: the topic "personal life" → tag `personal-life`.
- **Tags are intentionally duplicated:** in the frontmatter (`tags:`) — for search/graph/Dataview, and in the body (`**Hashtags:** #…`) — for readability. Keep them in sync.
- Timestamps: `MM:SS`, or `HH:MM:SS` once duration ≥ 1 h.
- **Language** — uppercase (`RU`/`EN`).
- **Topics:** each topic with its own timestamp in parentheses, ` · ` as the separator (for easy manual editing).
- **Long recordings (long-form):** if `duration_min >= long_form_from_min` (default 45) — render the extra `### Key Topics` section (a bulleted list of topics with timestamps and a brief gist from `key_topics`) and `### Decisions` (from `decisions`). Empty lists — don't render the section. Short recordings never have these sections.
- **Monologue:** omit roles — transcript lines look like `**[00:00]** …` with no "Speaker N". If a name is available (`--names`/voice-ID) — use the name.
- **Speaker wikilinks:** with `wikilink_speakers=true`/`--wikilink-speakers`, render named speakers as `[[Kate]]` (creates backlinks and graph nodes in Obsidian — useful later with voice-ID). **Off** by default (to avoid spawning empty notes); `Speaker 1/2` is NEVER turned into a wikilink.
- Key phrases can be **bolded** — but sparingly (don't highlight half the text). In v1 it's enough to bold only the `[time] Speaker:` prefixes; auto-highlighting within lines is optional and should stay conservative.
- **YAML safety:** escape values containing `:`/quotes (wrap in double quotes, double up internal quotes) — otherwise Obsidian won't parse the frontmatter. `source_file` is always quoted.
- In `--text` mode, omit the "Summary" block but keep the frontmatter (without `tags`, if there are no topics).

---

## 12. Resilience / resume / logs

- **Dedup by hash:** renaming a file does NOT trigger reprocessing (same hash → `done` in the manifest → skip). Changing the content → a new hash → process it.
- **Atomicity:** write `systems/raw/<hash>.json.tmp` and `out/<name>.md.tmp` first, then `os.replace(...)` to the final paths. Write the manifest the same way (tmp + `os.replace`).
- **Write order:** `in_progress` in the manifest → heavy work → save raw+md atomically → `done` in the manifest. This way a "half file marked done" can never happen.
- **Resume:** on startup, diff the folder against the manifest. Only process `to_do`, `in_progress` (unfinished), and — with `--retry-failed` — `failed`.
- **Graceful failure:** a failed file → `failed` + the error in the manifest and the per-file log + a stack trace; continue with the rest, don't bring down the whole process.
- **Logs:** a shared `logs/run.log` (run-level) + `logs/<name>__<hash8>.log` per file (§5.3). Protect the manifest with a `threading.Lock` (written from different workers).

---

## 13. Concurrency (staged pipeline)

There's a single GPU context → **Whisper itself is NOT parallelized across files** (Metal serializes it, and there's the memory budget). But stages of a different nature are pipelined:

```
[ffmpeg]  (CPU, --jobs workers) ──queue──▶ [ASR + diarize] (GPU, exactly 1 worker) ──queue──▶ [summary + render] (CPU/Ollama, --jobs workers)
```

- While file N is being processed on Metal, ffmpeg is preparing N+1, and Ollama is finishing off N−1.
- Implementation: `concurrent.futures.ThreadPoolExecutor` + a bounded `queue.Queue` (backpressure), or a simple 3-stage producer/consumer. The GPU stage is a single thread (serializing ASR+diarize so they don't fight over Metal/MPS).
- `--jobs` only controls the CPU stages. Default `jobs=3`.
- A `tqdm` progress bar over the file count + clear status lines.
- **Document the trade-off in the README:** why ASR isn't parallelized, and what the pipeline buys you.

---

## 14. Speaker names (`--names`) + groundwork for voice-ID

### v1 — implement now
- `--names "Me,Kate"` — a list of names in **order of the speaker's first appearance in the audio** (whoever speaks first gets the first name). Build the mapping `SPEAKER_00→Me`, `SPEAKER_01→Kate` from the timestamp of each speaker's first word. Extra/missing names — don't crash, leave `Speaker N` for uncovered speakers.

### Groundwork for later — do NOT implement, but don't block the path
The user wants to later automatically recognize people across different recordings (voice identification): say "this is Kate's voice" once, save the embedding, and the system then fills in the name automatically.

To make this possible without a rework later:
1. In the `diarize` stage, already **extract a per-speaker embedding** (an averaged vector over representative segments of that speaker; pyannote gives access to an embedding model) and put it in `speakers_meta[].embedding` in the raw JSON.
2. Create `voiceprints.py` with a **stub interface** and a `systems/voiceprints/` directory:
   ```python
   class VoiceprintStore:
       def enroll(self, name: str, embedding: list[float]) -> None: ...   # TODO: save to systems/voiceprints/<name>.json
       def identify(self, embedding: list[float], threshold: float = 0.5) -> str | None: ...  # TODO: nearest-neighbor cosine search
   ```
   In v1 the methods can be no-ops/`return None` with a `# TODO`. The important part is that embeddings are already being saved and the interface exists.
3. In `merge`, provide a hook: if `VoiceprintStore.identify()` returns a name — substitute it for `Speaker N` (in v1 it's always `None`, so behavior doesn't change).

---

## 15. Error handling and edge cases

- An empty/corrupt audio file → `failed` with a clear error, don't bring down the process.
- ffmpeg unavailable → a clear error at startup (check in `--warmup` and at the start of a run).
- Ollama not running / model not downloaded → a clear error + a hint to run `ollama pull qwen2.5:14b`. Not needed for `--text` mode — don't check the LLM there.
- A file with no speech (only silence/noise) → an empty transcript, note it in the log, Markdown with an empty transcript, don't crash.
- Very long recordings → map-reduce for the summary (§10.3). mlx-whisper's ASR chunks internally on its own.
- No HF token for pyannote → a clear error + a link to accept the model terms.

---

## 16. Implementation order (milestones) — follow the steps strictly

Implement incrementally; after every step there should be a runnable result.

1. **Skeleton:** `models.py` (dataclasses), `config.py`, `cli.py` (argparse with all the flags from §6, still stubs), `logging_setup.py`, `--dry-run` prints the plan. `setup_mac.sh`, `requirements.txt`, `config.example.toml`.
2. **ingest + manifest:** folder scan, blake2b, `manifest.py` (atomic write/read), `--dry-run` shows to_do/skip/redo. Dedup by hash.
3. **audio:** ffmpeg normalization + ffprobe duration, temp files in the system tmp directory.
4. **ASR (MLX):** `asr_mlx.py`, `--text` mode already gives a transcript with no roles and no LLM (title from the filename/default). Save the raw JSON.
5. **diarize + merge:** pyannote on MPS, speaker assignment, monologue collapse, `--speakers`/`--min`/`--max`, embedding extraction into `speakers_meta`.
6. **render:** the Markdown template from §11, filename sanitization/collisions (`naming.py`), `--rerender` mode from raw JSON.
7. **summarize (Ollama):** `summarize.py`, JSON output, timestamped topics, LLM title for technical filenames, map-reduce for long recordings. `--summary`, `--resummarize` modes.
8. **pipeline + concurrency:** the staged pipeline (§13), `--jobs`, progress bar, resume, `--retry-failed`, graceful failure.
9. **names + voiceprints groundwork:** `--names`, the `voiceprints.py` interface, the hook in `merge`.
10. **warmup, polish, README, tests.**

---

## 17. Acceptance criteria (verifiable)

The implementation is considered done when:

- [ ] `python -m transcriber --dry-run --input-folder ./audio` prints a correct plan without processing anything.
- [ ] Running against a folder creates one `out/YYYY-MM-DD — Title.md` per file, exactly per the §11 template.
- [ ] A file with a meaningful name → title from the filename; a file with a date/`REC_*`/`New Recording N` → title from the LLM, date from the filename/metadata.
- [ ] A monologue doesn't produce "Speaker 1/2"; a dialogue is correctly labeled with roles and timestamps.
- [ ] Language is detected automatically and written to the raw JSON, the log, and the Markdown.
- [ ] A repeat run reprocesses nothing (everything is `done` in the manifest); **renaming a file** does not trigger reprocessing; **changing its content** does.
- [ ] An interruption partway through (Ctrl-C / kill) doesn't leave a `done` entry for an unfinished file; a restart finishes it.
- [ ] `--resummarize` changes the summary without rerunning Whisper (verify via timing and logs — ASR is not invoked).
- [ ] `--text` gives a transcript without touching Ollama; `--summary` works from existing raw data without ASR.
- [ ] `--turbo`, `--only`, `--skip`, `--speakers`, `--names`, `--jobs`, `--retry-failed` all work as described.
- [ ] A failed file → `failed`, the rest still get processed; `--retry-failed` puts it back in the queue.
- [ ] `systems/raw/<hash>.json` contains a per-speaker `embedding` (non-empty for dialogues) — groundwork for voice-ID.
- [ ] `setup_mac.sh` installs everything from a clean machine and `--warmup` downloads/warms up the models.
- [ ] Summary length is adaptive: a short recording (<15 min) gives ~3–5 sentences, a long one (>90 min) gives more; the ranges are read from the `[summary].tiers` config (changeable without touching code). A recording ≥45 min additionally has "Key Topics" and "Decisions" sections; a short one doesn't.
- [ ] Every `.md` opens in Obsidian as a valid note: the frontmatter parses into Properties (date/language/speakers/duration/source_file/tags), tags show up in the tag pane, the filename has no forbidden Obsidian characters (`# ^ [ ] |`). The internal `systems/`/`logs/` folders never end up inside the vault's `out/` folder.

**Tests (`tests/`):** unit tests for pure logic, no models — `naming.py` (technical/meaningful name detection, iPhone-style names, date extraction, sanitization, collisions), `merge.py` (speaker assignment by overlap, monologue collapse by threshold), `manifest.py` (atomicity, dedup by hash, statuses), timestamp formatting. Mock ASR/diarize/LLM.

---

## 18. README (required)

Briefly: what it does, setup on an M4 (`setup_mac.sh` in one command), example invocations of every mode, flag reference, the `out/`/`systems/`/`logs/` layout, an "Architecture and trade-offs" section (single MLX+pyannote backend and why, raw JSON as the source of truth, the staged pipeline and why ASR isn't parallelized, groundwork for voice-ID).

**"Obsidian" section:** how to hook it up — point `--out` at a subfolder inside the vault; keep `systems/`/`logs/` outside the vault (or exclude them); notes arrive with frontmatter properties and tags, ready for Dataview/the graph; cover `--wikilink-speakers` and `--no-frontmatter`.
