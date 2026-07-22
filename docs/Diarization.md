# Diarization (`--diarize`) + voiceprint enrollment

Speaker diarization as its own deliverable, decoupled from summarization, plus
commands to seed the voiceprint store so `SPEAKER_00` becomes a stable name
 across recordings.

## `--diarize` mode

Fresh-audio pipeline: `normalize → ASR + diarize + merge + voiceprints →
render`. No summary, no pretty. Produces a raw doc with speakers and
`summary=None`, plus a speaker-tagged markdown transcript.

```bash
# transcript + speakers, no summary
python -m transcriber --diarize --input-folder ./audio

# name the speakers by dominance order (also auto-enrolls their voiceprints)
python -m transcriber --diarize --names "dad,kate"
```

Then finish the same raw docs separately:

```bash
python -m transcriber --summary          # fills the summary
python -m transcriber --pretty --summary # + LLM-cleaned transcript
```

### Why a separate mode

`full` bundles `diarize + summary`. Diarization (pyannote, ~1–2 GB) is only worth
it on real dialogue; on monologues `mono_threshold` collapses everyone into one
speaker, so it's paid work thrown away. `--diarize` lets you decide "speakers or
not" at transcription time — mass-run `--text` cheaply, run `--diarize` only on
recordings where "who said what" matters.

The speaker decision has to be made now, not later: diarization needs the
waveform, and a `--text` raw carries no speakers. A file already processed as
`--text` is `done` with no speakers, so `--diarize` will skip it — re-run it to
diarize. (Auto-detecting monologues from samples to skip pyannote entirely is
[issue #35](https://github.com/ValikProgrammer/RecordingsToMDOfflineTranscriber/issues/35).)

Resume: like `full`/`--text`, already-`done` content hashes are skipped.

## Voiceprints

Speaker embeddings are computed for free during diarization and stored on the raw
doc (`speakers_meta[*].embedding`). Matching a new speaker is cosine similarity
against a small per-name store under `systems/voiceprints/` — cheap, no extra
model. With voiceprints on (`voiceprint_enabled=true`), a named speaker in one
recording is recognized automatically in the next.

### Enroll from one recording — `--enroll NAME`

Takes the dominant speaker of each audio file in the input folder and enrolls it
under `NAME`. For clean single-speaker samples.

```bash
python -m transcriber --enroll "Dad" --input-folder ./samples/mama
```

### Enroll from a labeled multi-speaker raw — `--enroll-raw PATH|NAME`

Reads a raw doc whose speakers are already named (e.g. from `--diarize --names`,
or hand-corrected) and enrolls every named speaker's embedding at once. No audio,
no pyannote — it reuses the embeddings already in the raw. One labeled dialog
seeds several voices.

```bash
# by source-name / hash substring (searches systems/raw/*.json)
python -m transcriber --enroll-raw "dialog with mama"

# or a direct path to the raw JSON
python -m transcriber --enroll-raw systems/raw/ab12cd34.json
```

Speakers without a name, or without an embedding, are skipped.

## Code map

- `transcriber/cli.py` — `--diarize`/`--diarization`, `--enroll`, `--enroll-raw`
  flags; `resolve_mode` returns `"diarize"`.
- `transcriber/pipeline.py` — mode dispatch (`_safe_stage_b` runs diarize+merge,
  `_safe_stage_c` skips summary unless `full`); `resolve_raw_by_query` resolves
  `--enroll-raw` targets.
- `transcriber/stages/diarize.py` — pyannote wrapper + per-speaker embeddings.
- `transcriber/voiceprints.py` — `VoiceprintStore`, `enroll_named_speakers`,
  `identify_speakers`.
- `transcriber/__main__.py` — `cmd_enroll`, `cmd_enroll_from_raw`.
