# Rename stage — rename source audio too (addendum)

Date: 2026-07-22

Follow-up to `2026-07-21-rename-stage-design.md`. The original pass only renamed
the `.md` docs. The point was to rename the **source audio** as well, keeping the
recording and its note in sync. This addendum adds that, and restores the
algorithmic-date behaviour (frontmatter `Date:` priority) that regressed out of
`main`.

## Changes

### `--propose`
- Parse the doc's `Source file:` frontmatter (`parse_source_file`).
- Compute `new_audio_name` = same base as the new `.md` name, with the audio's
  own extension (`_audio_name_for`). Store `source_file` + `new_audio_name` in the
  plan alongside `new_title` / `new_name`.
- Date resolved algorithmically (restored): frontmatter `Date:` → filename date →
  file mtime (`resolve_date` / `parse_frontmatter_date`). LLM proposes title only.

### `--apply`
- Rename the source audio first (`_rename_audio`, collision-safe) so the md's
  `Source file:` can point at the final name; then rewrite `Title:` / `# …` /
  `Source file:` and rename the `.md` + `out/pretty/` twin.
- Missing audio → warn, still rename the `.md`, leave `Source file:` as-is.
- Sync `systems/manifest.json` (`update_manifest`): match the entry by `out_path`
  filename, update `out_path` + `source_name`. Best-effort; dedup is by content
  hash so this is cosmetic. `--no-manifest` skips it.
- New flags: `--audio-folder` (default config `input_folder`), `--no-manifest`.

## Reuse

`naming.resolve_collision`, `render.yaml_escape`, `config.input_folder` /
`systems_folder`, `manifest.Manifest` (`all_entries` / `upsert`).

## Tests (added to `tests/test_rename.py`)

- `parse_source_file`; `new_audio_name` matches the `.md` base with audio ext.
- apply: audio renamed + `Source file:` synced; missing-audio path still renames
  the `.md` and leaves `Source file:` unchanged.
- `update_manifest` updates `out_path` + `source_name` by out_path match.

## Verification

1. `pytest tests/test_rename.py -q`; full `pytest -q`.
2. Manual on `out/` + `audio/` (Ollama running): `--classify` → `--propose`
   (check `new_audio_name`) → `--apply` → confirm audio + `.md` + pretty renamed,
   `Source file:` and manifest in sync.
