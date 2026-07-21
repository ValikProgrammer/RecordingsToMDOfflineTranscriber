from pathlib import Path

from transcriber import trim


def test_parse_silencedetect_pairs_and_drops_dangling_start():
    stderr = (
        "[silencedetect @ 0x1] silence_start: 0\n"
        "[silencedetect @ 0x1] silence_end: 120.5 | silence_duration: 120.5\n"
        "size=... time=...\n"
        "[silencedetect @ 0x1] silence_start: 3400.0\n"
        "[silencedetect @ 0x1] silence_end: 3600.0 | silence_duration: 200\n"
        "[silencedetect @ 0x1] silence_start: 7000.0\n"  # no matching end -> dropped
    )
    assert trim.parse_silencedetect(stderr) == [(0.0, 120.5), (3400.0, 3600.0)]


def test_plan_file_filters_by_min_gap_and_clamps():
    silences = [(-1.0, 30.0), (100.0, 105.0), (200.0, 260.0)]  # 5s one is below min_gap
    plan = trim.plan_file(silences, duration=300.0, min_gap_sec=20.0, min_total_sec=60.0)
    assert plan["action"] == "trim"
    assert plan["cuts"] == [[0.0, 30.0], [200.0, 260.0]]
    assert plan["total_cut_sec"] == 90.0


def test_plan_file_skips_when_total_below_min_total():
    silences = [(0.0, 25.0)]
    plan = trim.plan_file(silences, duration=300.0, min_gap_sec=20.0, min_total_sec=60.0)
    assert plan["action"] == "skip"
    assert plan["cuts"] == []


def test_keep_ranges_is_complement_of_cuts():
    assert trim.keep_ranges([[0.0, 100.0], [500.0, 600.0]], duration=1000.0) == [
        (100.0, 500.0),
        (600.0, 1000.0),
    ]


def test_keep_ranges_no_cuts_keeps_whole():
    assert trim.keep_ranges([], duration=50.0) == [(0.0, 50.0)]


def test_aselect_expr_builds_between_union():
    expr = trim._aselect_expr([(100.0, 500.0), (600.0, 1000.0)])
    assert expr == "between(t,100.000,500.000)+between(t,600.000,1000.000)"


def test_apply_plan_only_trims_marked_files(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(trim, "apply_cut", lambda src, cuts, dur, dst: calls.append(dst.name))

    (tmp_path / "a.m4a").write_bytes(b"x")
    (tmp_path / "b.m4a").write_bytes(b"x")
    plan = {
        "files": [
            {"file": "a.m4a", "duration_sec": 100.0, "action": "trim", "cuts": [[0.0, 30.0]]},
            {"file": "b.m4a", "duration_sec": 100.0, "action": "skip", "cuts": []},
            {"file": "missing.m4a", "duration_sec": 100.0, "action": "trim", "cuts": [[0.0, 30.0]]},
        ]
    }
    applied = trim.apply_plan(plan, tmp_path, tmp_path / "edited", log=lambda *_: None)
    assert applied == 1
    assert calls == ["a.m4a"]


def test_main_apply_missing_plan_returns_1(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = trim.main(["--apply", "--plan", "nope.json"])
    assert rc == 1
    assert "not found" in capsys.readouterr().out.lower()
