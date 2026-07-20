import logging
import sys
import types

from transcriber.config import Config
from transcriber.models import RawDoc, Segment
from transcriber.stages.pretty import format_transcript_for_pretty, render_pretty_transcript

LOG = logging.getLogger("test")


def _doc(segments, language="ru"):
    return RawDoc(
        schema=1, content_hash="h", source_name="a.m4a", source_path="/a.m4a",
        language=language, duration_sec=120.0, num_speakers=2, is_monologue=False,
        asr=None, created_at="", segments=segments, speakers_meta=[], summary=None,
    )


def test_format_includes_speaker_and_timecode():
    doc = _doc([Segment(65.0, 70.0, "Галя", "привет")])
    assert format_transcript_for_pretty(doc) == ["[01:05] Галя: привет"]


def test_format_omits_speaker_when_none():
    doc = _doc([Segment(0.0, 2.0, None, "текст")])
    assert format_transcript_for_pretty(doc) == ["[00:00] текст"]


def _install_fake_ollama(monkeypatch, content):
    fake = types.ModuleType("ollama")
    fake.chat = lambda model, messages, **kw: {"message": {"content": content}}
    monkeypatch.setitem(sys.modules, "ollama", fake)


def test_render_pretty_transcript_returns_llm_text(monkeypatch):
    _install_fake_ollama(monkeypatch, "  PRETTY VERSION  ")
    doc = _doc([Segment(0.0, 2.0, "Галя", "привет")])
    assert render_pretty_transcript(doc, Config(), LOG) == "PRETTY VERSION"
