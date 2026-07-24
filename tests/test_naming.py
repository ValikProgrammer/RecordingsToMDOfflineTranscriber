from datetime import date
from pathlib import Path

import pytest

from transcriber import naming


@pytest.mark.parametrize(
    "stem",
    ["2026-07-12", "20260712", "2026.07.12", "12-07-2026"],
)
def test_pure_date_is_technical(stem):
    assert naming.is_technical_name(stem)


@pytest.mark.parametrize(
    "stem",
    ["REC_20260712", "AUD-20260712", "voice 001", "recording_003"],
)
def test_recorder_prefix_is_technical(stem):
    assert naming.is_technical_name(stem)


@pytest.mark.parametrize(
    "stem",
    [
        "New Recording",
        "New Recording 12",
        "Новая запись",
        "Новая запись 5",
        "Recording 3",
        "Запись 7",
        "новая запись",
        "ЗАПИСЬ 1",
    ],
)
def test_iphone_style_is_technical_case_insensitive(stem):
    assert naming.is_technical_name(stem)


@pytest.mark.parametrize("stem", ["audio", "voice memo", "___", "12345", "1_2_3"])
def test_junk_names_are_technical(stem):
    assert naming.is_technical_name(stem)


@pytest.mark.parametrize("stem", ["team call notes", "project sync call", "client interview"])
def test_meaningful_names_are_not_technical(stem):
    assert not naming.is_technical_name(stem)


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("2026-07-12", date(2026, 7, 12)),
        ("REC_20260712", date(2026, 7, 12)),
        ("12-07-2026", date(2026, 7, 12)),
        ("team call notes", None),
        # month-name device auto-names: no year -> default 2025
        ("Dec 6, 23.57", date(2025, 12, 6)),
        ("Sep 2, 16.23 ИстФак дечоки", date(2025, 9, 2)),
        ("Dec 16, 17.22 рассказываю про Грузию", date(2025, 12, 16)),
        ("December 6", date(2025, 12, 6)),
        # explicit year in the name is honored
        ("Dec 6, 2024", date(2024, 12, 6)),
    ],
)
def test_extract_date_from_name(stem, expected):
    assert naming.extract_date_from_name(stem) == expected


def test_extract_date_from_file_uses_birthtime_or_mtime(tmp_path):
    path = tmp_path / "audio.m4a"
    path.write_bytes(b"x")
    result = naming.extract_date_from_file(path)
    assert isinstance(result, date)


def test_extract_creation_time_none_for_non_media(tmp_path):
    path = tmp_path / "notmedia.m4a"
    path.write_bytes(b"not audio")
    assert naming.extract_creation_time(path) is None


def _touch_mtime(path, day: date):
    import os, time
    ts = time.mktime(day.timetuple())
    os.utime(path, (ts, ts))


def test_resolve_date_takes_oldest_across_sources(tmp_path):
    # name -> 2025-12-06, creation -> 2026-01-10, mtime -> recent: oldest wins
    path = tmp_path / "Dec 6, 23.57.m4a"
    path.write_bytes(b"x")
    _touch_mtime(path, date(2026, 7, 20))
    assert naming.resolve_date("Dec 6, 23.57", path, creation_time=date(2026, 1, 10)) == date(2025, 12, 6)


def test_resolve_date_uses_creation_when_name_has_no_date(tmp_path):
    path = tmp_path / "Natalia talk.m4a"
    path.write_bytes(b"x")
    _touch_mtime(path, date(2026, 7, 20))
    assert naming.resolve_date("Natalia talk", path, creation_time=date(2026, 6, 18)) == date(2026, 6, 18)


def test_resolve_date_conflict_prefers_oldest(tmp_path):
    # name has a full date but creation is older -> oldest (creation) wins
    path = tmp_path / "2026-07-12.m4a"
    path.write_bytes(b"x")
    _touch_mtime(path, date(2026, 7, 20))
    assert naming.resolve_date("2026-07-12", path, creation_time=date(2025, 1, 1)) == date(2025, 1, 1)


def test_resolve_date_falls_back_to_filesystem(tmp_path):
    path = tmp_path / "no date here.m4a"
    path.write_bytes(b"x")
    _touch_mtime(path, date(2024, 3, 3))
    assert naming.resolve_date("no date here", path, creation_time=None) == date(2024, 3, 3)


def test_normalize_title_from_name_truncates_and_capitalizes():
    assert naming.normalize_title_from_name("team-call_with_Jamie.about.project.notes") == "Team call with Jamie"


def test_normalize_title_from_name_empty_falls_back():
    assert naming.normalize_title_from_name("___") == "Recording"


def test_default_title_for_date():
    assert naming.default_title_for_date(date(2026, 7, 12)) == "Recording 2026-07-12"


def test_sanitize_filename_component_strips_forbidden_chars():
    dirty = 'Title: "quoted" / \\ * ? < > | # ^ [1]'
    cleaned = naming.sanitize_filename_component(dirty)
    for ch in '\\/:*?"<>|#^[]':
        assert ch not in cleaned


def test_sanitize_filename_component_truncates_length():
    long_title = "Word " * 30
    cleaned = naming.sanitize_filename_component(long_title, max_len=60)
    assert len(cleaned) <= 60


def test_sanitize_filename_component_keeps_dash_separator():
    assert naming.sanitize_filename_component("Conversation — About Cyprus") == "Conversation — About Cyprus"


def test_build_output_filename_format():
    name = naming.build_output_filename(date(2026, 7, 12), "Call with Jamie")
    assert name == "2026-07-12 — Call with Jamie.md"


def test_resolve_collision_returns_bare_path_when_free(tmp_path):
    result = naming.resolve_collision(tmp_path, "note.md")
    assert result == tmp_path / "note.md"


def test_resolve_collision_appends_incrementing_suffix(tmp_path):
    (tmp_path / "note.md").write_text("x")
    (tmp_path / "note (2).md").write_text("x")
    result = naming.resolve_collision(tmp_path, "note.md")
    assert result == tmp_path / "note (3).md"


def test_resolve_collision_reserves_the_path(tmp_path):
    # reservation is what makes it race-safe: the returned path must now exist
    result = naming.resolve_collision(tmp_path, "note.md")
    assert result.exists()


def test_resolve_collision_hands_out_unique_paths_under_concurrency(tmp_path):
    import threading

    results: list[Path] = []
    lock = threading.Lock()
    barrier = threading.Barrier(20)

    def grab():
        barrier.wait()  # maximize overlap on the check-then-create window
        p = naming.resolve_collision(tmp_path, "note.md")
        with lock:
            results.append(p)

    threads = [threading.Thread(target=grab) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 20
    assert len(set(results)) == 20  # no two callers got the same path


def test_transliterate_ru_matches_spec_example():
    assert naming.transliterate_ru("рабочая встреча") == "rabochaya vstrecha"


def test_slugify_for_log_matches_spec_example():
    assert naming.slugify_for_log("рабочая встреча") == "rabochaya-vstrecha"


def test_slugify_for_log_never_empty():
    assert naming.slugify_for_log("") == "file"
    assert naming.slugify_for_log("!!!") == "file"


def test_slugify_tag_keeps_cyrillic_and_hyphenates():
    assert naming.slugify_tag("личная жизнь") == "личная-жизнь"


def test_slugify_tag_strips_hash_and_lowercases():
    assert naming.slugify_tag("#Telegram") == "telegram"


def test_slugify_tag_drops_numeric_only():
    assert naming.slugify_tag("2026") is None


def test_slugify_tag_drops_empty():
    assert naming.slugify_tag("   ") is None
