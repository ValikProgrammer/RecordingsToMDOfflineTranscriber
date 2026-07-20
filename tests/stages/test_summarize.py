import json
import logging
import sys
import types

from transcriber.config import Config, SummaryTier
from transcriber.models import AsrInfo, RawDoc, Segment
from transcriber.stages.summarize import (
    call_ollama_json,
    chunk_transcript,
    format_transcript_with_timestamps,
    parse_ts_hint,
    select_tier,
    summarize,
)

LOG = logging.getLogger("test")


def _doc(duration_sec=300.0, segments=None) -> RawDoc:
    return RawDoc(
        schema=1,
        content_hash="blake2b:x",
        source_name="a.m4a",
        source_path="/a.m4a",
        language="ru",
        duration_sec=duration_sec,
        num_speakers=1,
        is_monologue=True,
        asr=AsrInfo(backend="mlx", model="large-v3", turbo=False),
        created_at="2026-07-19T21:00:00Z",
        segments=segments if segments is not None else [Segment(0.0, 3.0, None, "hello world")],
    )


def test_select_tier_picks_first_matching():
    tiers = [SummaryTier(15, "3-5"), SummaryTier(45, "5-8"), SummaryTier(100000, "10-15")]
    assert select_tier(600.0, tiers).sentences == "3-5"  # 10 min


def test_select_tier_falls_back_to_last_when_longer_than_all():
    tiers = [SummaryTier(15, "3-5"), SummaryTier(45, "5-8")]
    assert select_tier(999999.0, tiers).sentences == "5-8"


def test_format_transcript_with_timestamps():
    doc = _doc(segments=[Segment(65.0, 70.0, None, "hello")])
    assert format_transcript_with_timestamps(doc) == ["[01:05] hello"]


def test_chunk_transcript_respects_max_chars():
    lines = ["a" * 10, "b" * 10, "c" * 10]
    chunks = chunk_transcript(lines, max_chars=15)
    assert chunks == [["a" * 10], ["b" * 10], ["c" * 10]]


def test_chunk_transcript_packs_lines_under_limit():
    lines = ["a" * 5, "b" * 5, "c" * 5]
    chunks = chunk_transcript(lines, max_chars=15)
    assert chunks == [["a" * 5, "b" * 5, "c" * 5]]


def test_parse_ts_hint_mmss_and_hhmmss():
    doc = _doc(segments=[])
    assert parse_ts_hint("01:05", doc) == 65.0
    assert parse_ts_hint("01:00:05", doc) == 3605.0


def test_parse_ts_hint_snaps_to_nearest_segment():
    doc = _doc(segments=[Segment(10.0, 12.0, None, "a"), Segment(100.0, 102.0, None, "b")])
    assert parse_ts_hint("00:11", doc) == 10.0
    assert parse_ts_hint("01:40", doc) == 100.0


def _install_fake_ollama(monkeypatch, responses):
    fake_module = types.ModuleType("ollama")
    calls = []

    def fake_chat(model, format, messages):
        calls.append({"model": model, "format": format, "messages": messages})
        content = responses[len(calls) - 1] if len(calls) <= len(responses) else responses[-1]
        return {"message": {"content": content}}

    fake_module.chat = fake_chat
    monkeypatch.setitem(sys.modules, "ollama", fake_module)
    return calls


def test_call_ollama_json_parses_valid_response(monkeypatch):
    _install_fake_ollama(monkeypatch, [json.dumps({"title": "T"})])
    result = call_ollama_json("qwen2.5:14b", "sys", "user", LOG)
    assert result == {"title": "T"}


def test_call_ollama_json_retries_once_then_succeeds(monkeypatch):
    calls = _install_fake_ollama(monkeypatch, ["not json", json.dumps({"title": "ok"})])
    result = call_ollama_json("qwen2.5:14b", "sys", "user", LOG)
    assert result == {"title": "ok"}
    assert len(calls) == 2


def test_call_ollama_json_raises_after_two_bad_responses(monkeypatch):
    _install_fake_ollama(monkeypatch, ["not json", "still not json"])
    try:
        call_ollama_json("qwen2.5:14b", "sys", "user", LOG)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_summarize_single_chunk_builds_summary_with_tiers_and_topics(monkeypatch):
    response = json.dumps(
        {
            "title": "Note",
            "summary": "Short summary",
            "topics": [{"term": "budget", "ts_hint": "00:05"}],
            "hashtags": ["budget", "2026"],
            "key_topics": [],
            "decisions": [],
        }
    )
    calls = _install_fake_ollama(monkeypatch, [response])
    doc = _doc(duration_sec=300.0, segments=[Segment(5.0, 8.0, None, "about the budget")])
    cfg = Config(llm_ctx=8192)

    summary = summarize(doc, cfg, LOG)

    assert summary.title == "Note"
    assert summary.text == "Short summary"
    assert summary.topics[0].term == "budget"
    assert summary.topics[0].ts == 5.0
    assert summary.hashtags == ["budget", "2026"]
    assert summary.generated is True
    assert summary.llm_model == "qwen2.5:14b"
    assert len(calls) == 1


def test_summarize_long_transcript_triggers_map_reduce(monkeypatch):
    partial = json.dumps({"title": "T", "summary": "S", "topics": [], "hashtags": [], "key_topics": [], "decisions": []})
    reduced = json.dumps({"title": "Final", "summary": "Combined", "topics": [], "hashtags": [], "key_topics": [], "decisions": []})
    calls = _install_fake_ollama(monkeypatch, [partial, partial, reduced])

    # Force exactly 2 chunks regardless of chunk_transcript's char-packing details
    # (already covered by its own tests above) so this test only exercises the
    # map-reduce call count: 2 map chunks + 1 reduce = 3 ollama.chat calls.
    monkeypatch.setattr(
        "transcriber.stages.summarize.chunk_transcript",
        lambda lines, max_chars: [["[00:00] part one"], ["[10:00] part two"]],
    )

    doc = _doc(duration_sec=3600.0, segments=[Segment(0.0, 1.0, None, "text")])
    cfg = Config(llm_ctx=10)

    summary = summarize(doc, cfg, LOG)

    assert summary.title == "Final"
    assert summary.text == "Combined"
    assert len(calls) == 3  # 2 map chunks + 1 reduce
