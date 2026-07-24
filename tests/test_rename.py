import json
import logging
import sys
import types
from pathlib import Path

from transcriber import rename

LOG = logging.getLogger("test")


# --- excel_label -------------------------------------------------------------

def test_excel_label_sequence():
    assert rename.excel_label(0) == "A"
    assert rename.excel_label(25) == "Z"
    assert rename.excel_label(26) == "AA"
    assert rename.excel_label(27) == "AB"
    assert rename.excel_label(51) == "AZ"
    assert rename.excel_label(52) == "BA"


# --- .md parsing -------------------------------------------------------------

_DOC = """---
Title: "Dec 6, 23 57"
Date: 2026-07-20
Language: RU
Source file: "Dec 6, 23.57.m4a"
---

# Dec 6, 23 57

### Summary
Обсуждались документы за границу и апостиль. Встреча с друзьями.

**Topics:**
- [01:15] Запись на апостиль
- [07:08] Контакты и помощь друзей

**Hashtags:** #апостиль
"""


def test_parse_summary_and_topics():
    summary, topics = rename.parse_summary_and_topics(_DOC)
    assert "апостиль" in summary
    assert topics == ["Запись на апостиль", "Контакты и помощь друзей"]


def test_parse_handles_missing_topics():
    text = "### Summary\nПросто саммари.\n"
    summary, topics = rename.parse_summary_and_topics(text)
    assert summary == "Просто саммари."
    assert topics == []


# --- fake ollama -------------------------------------------------------------

def _install_fake_ollama(monkeypatch, responses):
    fake_module = types.ModuleType("ollama")
    calls = []

    def fake_chat(model, format, messages):
        calls.append({"model": model, "messages": messages})
        content = responses[len(calls) - 1] if len(calls) <= len(responses) else responses[-1]
        return {"message": {"content": content}}

    fake_module.chat = fake_chat
    monkeypatch.setitem(sys.modules, "ollama", fake_module)
    return calls


# --- classify ----------------------------------------------------------------

def test_classify_maps_labels_back_to_names(monkeypatch):
    _install_fake_ollama(monkeypatch, [json.dumps({"rename": ["A", "C"]})])
    names = ["junk1.md", "good.md", "junk2.md"]
    decisions = rename.classify_files(names, "m", LOG)
    assert decisions == {"junk1.md": True, "good.md": False, "junk2.md": True}


def test_classify_batches_reset_labels(monkeypatch):
    calls = _install_fake_ollama(
        monkeypatch, [json.dumps({"rename": ["A"]}), json.dumps({"rename": ["B"]})]
    )
    names = ["a.md", "b.md", "c.md", "d.md"]
    decisions = rename.classify_files(names, "m", LOG, batch_size=2)
    assert len(calls) == 2
    # batch1 A->a (rename), b keep; batch2 A->c keep, B->d rename
    assert decisions == {"a.md": True, "b.md": False, "c.md": False, "d.md": True}


# --- propose -----------------------------------------------------------------

def test_propose_returns_titles(monkeypatch):
    _install_fake_ollama(monkeypatch, [json.dumps({"A": "Разговор о документах"})])
    entries = [{"name": "junk.md", "summary": "s", "topics": ["t"]}]
    titles = rename.propose_titles(entries, "m", LOG)
    assert titles == {"junk.md": "Разговор о документах"}


# --- title rewrite -----------------------------------------------------------

def test_rewrite_title_frontmatter_and_heading():
    out = rename.rewrite_title(_DOC, "Новый заголовок", has_frontmatter=True)
    assert 'Title: "Новый заголовок"' in out
    assert "# Новый заголовок" in out
    assert "Dec 6" not in out.splitlines()[1]  # old Title line replaced


def test_rewrite_title_heading_only_for_pretty():
    pretty = "# Dec 6, 23 57\n\n[00:00] SPEAKER_04: hi\n"
    out = rename.rewrite_title(pretty, "Новый", has_frontmatter=False)
    assert out.startswith("# Новый\n")


# --- apply -------------------------------------------------------------------

def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_apply_entry_renames_and_rewrites_twin(tmp_path):
    folder = tmp_path / "out"
    old = "2026-07-20 — Dec 6, 23 57.md"
    _write(folder / old, _DOC)
    _write(folder / "pretty" / old, "# Dec 6, 23 57\n\n[00:00] hi\n")

    entry = {
        "file": old,
        "action": "rename",
        "new_title": "Документы за границу",
        "new_name": "2026-07-20 — Документы за границу.md",
    }
    audio = tmp_path / "audio"
    audio.mkdir()
    assert rename.apply_entry(entry, folder, "pretty", audio, log=lambda *a: None) is not None

    new = folder / "2026-07-20 — Документы за границу.md"
    assert new.exists()
    assert not (folder / old).exists()
    assert 'Title: "Документы за границу"' in new.read_text(encoding="utf-8")

    pnew = folder / "pretty" / "2026-07-20 — Документы за границу.md"
    assert pnew.exists()
    assert pnew.read_text(encoding="utf-8").startswith("# Документы за границу\n")


def test_apply_plan_skips_keep_and_counts_renames(tmp_path):
    folder = tmp_path / "out"
    _write(folder / "2026-07-20 — junk.md", _DOC)
    _write(folder / "2026-07-20 — Good name.md", _DOC)
    plan = {
        "folder": str(folder),
        "files": [
            {"file": "2026-07-20 — junk.md", "action": "rename",
             "new_title": "Хорошее имя", "new_name": "2026-07-20 — Хорошее имя.md"},
            {"file": "2026-07-20 — Good name.md", "action": "keep"},
        ],
    }
    applied = rename.apply_plan(plan, folder, "pretty", tmp_path / "audio", log=lambda *a: None)
    assert applied == 1
    assert (folder / "2026-07-20 — Хорошее имя.md").exists()
    assert (folder / "2026-07-20 — Good name.md").exists()


def test_apply_entry_collision_does_not_clobber(tmp_path):
    folder = tmp_path / "out"
    old = "2026-07-20 — junk.md"
    _write(folder / old, _DOC)
    _write(folder / "2026-07-20 — Taken.md", "existing\n")
    entry = {
        "file": old, "action": "rename",
        "new_title": "Taken", "new_name": "2026-07-20 — Taken.md",
    }
    rename.apply_entry(entry, folder, "pretty", tmp_path / "audio", log=lambda *a: None)
    # original "Taken" file untouched
    assert (folder / "2026-07-20 — Taken.md").read_text(encoding="utf-8") == "existing\n"
    # renamed file landed on a collision-suffixed path
    assert (folder / "2026-07-20 — Taken (2).md").exists()


# --- plan building integration ----------------------------------------------

def test_build_classify_plan_writes_actions(monkeypatch, tmp_path):
    folder = tmp_path / "out"
    _write(folder / "2026-07-20 — Dec 6.md", _DOC)
    _write(folder / "2026-07-20 — Natalia scheduler.md", _DOC)
    _install_fake_ollama(monkeypatch, [json.dumps({"rename": ["A"]})])
    plan = rename.build_classify_plan(folder, "m", LOG, batch_size=576)
    actions = {e["file"]: e["action"] for e in plan["files"]}
    assert actions["2026-07-20 — Dec 6.md"] == "rename"
    assert actions["2026-07-20 — Natalia scheduler.md"] == "keep"


def test_parse_frontmatter_date():
    from datetime import date

    assert rename.parse_frontmatter_date(_DOC) == date(2026, 7, 20)
    assert rename.parse_frontmatter_date("# no frontmatter\n") is None


def test_fill_proposals_sets_new_name_from_date_in_filename(monkeypatch, tmp_path):
    folder = tmp_path / "out"
    _write(folder / "2026-07-20 — Dec 6.md", _DOC)
    _install_fake_ollama(monkeypatch, [json.dumps({"A": "Документы за границу"})])
    plan = {
        "folder": str(folder),
        "files": [{"file": "2026-07-20 — Dec 6.md", "action": "rename", "reason": "x"}],
    }
    rename.fill_proposals(plan, folder, "m", LOG, batch_size=576)
    e = plan["files"][0]
    assert e["new_title"] == "Документы за границу"
    assert e["new_name"] == "2026-07-20 — Документы за границу.md"


def test_fill_proposals_prefers_frontmatter_date_over_filename(monkeypatch, tmp_path):
    # filename says May 01, frontmatter (Obsidian) says Jul 20 -> frontmatter wins
    folder = tmp_path / "out"
    _write(folder / "2026-05-01 — junk.md", _DOC)  # _DOC frontmatter Date is 2026-07-20
    _install_fake_ollama(monkeypatch, [json.dumps({"A": "Настоящее имя"})])
    plan = {
        "folder": str(folder),
        "files": [{"file": "2026-05-01 — junk.md", "action": "rename", "reason": "x"}],
    }
    rename.fill_proposals(plan, folder, "m", LOG, batch_size=576)
    assert plan["files"][0]["new_name"] == "2026-07-20 — Настоящее имя.md"


# --- date resolution prefers the audio file over a poisoned frontmatter Date ---

_POISONED_DOC = """---
Title: "Sep 24, 11 20"
Date: 2026-07-20
Language: RU
Source file: "Sep 24, 11.20.m4a"
---

# Sep 24, 11 20

### Summary
Тестовое саммари.
"""


def test_resolve_date_prefers_audio_file_over_poisoned_frontmatter(tmp_path):
    # frontmatter Date (2026-07-20) is the very bug being fixed -- it's the
    # processing date the .md happened to be generated on, not the recording
    # date. When the source audio is available, its own name/metadata/fs
    # signals (oldest-of-all, per naming.resolve_date) are more trustworthy.
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "Sep 24, 11.20.m4a").write_bytes(b"x")
    day = rename.resolve_date(
        "2026-07-20 — Sep 24.md", tmp_path, _POISONED_DOC, audio_folder=audio_dir
    )
    from datetime import date

    assert day == date(2025, 9, 24)


def test_resolve_date_falls_back_when_audio_missing(tmp_path):
    from datetime import date

    day = rename.resolve_date(
        "2026-05-01 — junk.md", tmp_path, _DOC, audio_folder=tmp_path / "audio"
    )
    assert day == date(2026, 7, 20)  # falls back to frontmatter Date from _DOC


def test_resolve_date_without_audio_folder_uses_old_priority(tmp_path):
    from datetime import date

    day = rename.resolve_date("2026-05-01 — junk.md", tmp_path, _DOC)
    assert day == date(2026, 7, 20)


def test_fill_proposals_passes_audio_folder_to_resolve_date(monkeypatch, tmp_path):
    folder = tmp_path / "out"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "Sep 24, 11.20.m4a").write_bytes(b"x")
    _write(folder / "2026-07-20 — Sep 24.md", _POISONED_DOC)
    _install_fake_ollama(monkeypatch, [json.dumps({"A": "Истфак"})])
    plan = {
        "folder": str(folder),
        "files": [{"file": "2026-07-20 — Sep 24.md", "action": "rename", "reason": "x"}],
    }
    rename.fill_proposals(plan, folder, "m", LOG, batch_size=576, audio_folder=audio_dir)
    assert plan["files"][0]["new_name"] == "2025-09-24 — Истфак.md"


# --- source audio ------------------------------------------------------------

def test_parse_source_file():
    assert rename.parse_source_file(_DOC) == "Dec 6, 23.57.m4a"
    assert rename.parse_source_file("# no frontmatter\n") is None


def test_fill_proposals_sets_audio_name_matching_md(monkeypatch, tmp_path):
    folder = tmp_path / "out"
    _write(folder / "2026-07-20 — Dec 6.md", _DOC)  # Source file: Dec 6, 23.57.m4a
    _install_fake_ollama(monkeypatch, [json.dumps({"A": "Документы за границу"})])
    plan = {
        "folder": str(folder),
        "files": [{"file": "2026-07-20 — Dec 6.md", "action": "rename", "reason": "x"}],
    }
    rename.fill_proposals(plan, folder, "m", LOG, batch_size=576)
    e = plan["files"][0]
    assert e["source_file"] == "Dec 6, 23.57.m4a"
    # same base as the .md, but the audio extension
    assert e["new_audio_name"] == "2026-07-20 — Документы за границу.m4a"


def test_apply_entry_renames_audio_and_syncs_source_file(tmp_path):
    folder = tmp_path / "out"
    audio = tmp_path / "audio"
    old = "2026-07-20 — Dec 6.md"
    _write(folder / old, _DOC)
    _write(audio / "Dec 6, 23.57.m4a", "AUDIO")

    entry = {
        "file": old, "action": "rename",
        "new_title": "Документы за границу",
        "new_name": "2026-07-20 — Документы за границу.md",
        "source_file": "Dec 6, 23.57.m4a",
        "new_audio_name": "2026-07-20 — Документы за границу.m4a",
    }
    result = rename.apply_entry(entry, folder, "pretty", audio, log=lambda *a: None)

    new_audio = audio / "2026-07-20 — Документы за границу.m4a"
    assert new_audio.exists()
    assert new_audio.read_text(encoding="utf-8") == "AUDIO"
    assert not (audio / "Dec 6, 23.57.m4a").exists()
    assert result["new_audio"] == "2026-07-20 — Документы за границу.m4a"
    # md's Source file: now points at the renamed audio
    md = (folder / "2026-07-20 — Документы за границу.md").read_text(encoding="utf-8")
    assert 'Source file: "2026-07-20 — Документы за границу.m4a"' in md


def test_apply_entry_missing_audio_still_renames_md(tmp_path):
    folder = tmp_path / "out"
    audio = tmp_path / "audio"
    audio.mkdir()  # empty — audio not present
    old = "2026-07-20 — Dec 6.md"
    _write(folder / old, _DOC)
    entry = {
        "file": old, "action": "rename",
        "new_title": "Документы за границу",
        "new_name": "2026-07-20 — Документы за границу.md",
        "source_file": "Dec 6, 23.57.m4a",
        "new_audio_name": "2026-07-20 — Документы за границу.m4a",
    }
    result = rename.apply_entry(entry, folder, "pretty", audio, log=lambda *a: None)
    assert result["new_audio"] is None
    md = (folder / "2026-07-20 — Документы за границу.md").read_text(encoding="utf-8")
    assert (folder / "2026-07-20 — Документы за границу.md").exists()
    # Source file untouched since audio wasn't renamed
    assert 'Source file: "Dec 6, 23.57.m4a"' in md


def test_update_manifest_syncs_out_path_and_source_name(tmp_path):
    from transcriber.manifest import Manifest
    from transcriber.models import ManifestEntry

    mpath = tmp_path / "manifest.json"
    m = Manifest(mpath)
    m.upsert(ManifestEntry(
        content_hash="blake2b:x",
        source_name="Dec 6, 23.57.m4a",
        status="done",
        out_path="out/2026-07-20 — Dec 6.md",
    ))
    result = {
        "old_md": "2026-07-20 — Dec 6.md",
        "new_md": "2026-07-20 — Документы за границу.md",
        "new_audio": "2026-07-20 — Документы за границу.m4a",
    }
    assert rename.update_manifest(m, result, log=lambda *a: None) is True

    reloaded = Manifest(mpath).get("blake2b:x")
    assert reloaded.out_path == "out/2026-07-20 — Документы за границу.md"
    assert reloaded.source_name == "2026-07-20 — Документы за границу.m4a"
