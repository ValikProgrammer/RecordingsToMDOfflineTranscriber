import logging
import sys
import types

from transcriber.stages.asr_mlx import (
    DEFAULT_INITIAL_PROMPT,
    FULL_REPO,
    TURBO_REPO,
    build_initial_prompt,
    transcribe,
)

LOG = logging.getLogger("test")


def _install_fake_mlx_whisper(monkeypatch, fixture, capture=None):
    fake_module = types.ModuleType("mlx_whisper")

    def fake_transcribe(path, **kwargs):
        if capture is not None:
            capture["path"] = path
            capture["repo"] = kwargs.get("path_or_hf_repo")
            capture["word_timestamps"] = kwargs.get("word_timestamps")
            capture.update(kwargs)
        return fixture

    fake_module.transcribe = fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_module)


def test_transcribe_converts_segments_and_words(monkeypatch, tmp_path):
    fixture = {
        "language": "ru",
        "segments": [
            {
                "start": 0.0,
                "end": 1.2,
                "text": " hello ",
                "words": [{"word": " hello", "start": 0.0, "end": 1.2}],
            }
        ],
    }
    _install_fake_mlx_whisper(monkeypatch, fixture)

    result = transcribe(tmp_path / "a.wav", turbo=False, log=LOG)

    assert result.language == "ru"
    assert result.backend == "mlx"
    assert result.model == "large-v3"
    assert result.turbo is False
    assert result.segments[0].text == "hello"
    assert result.segments[0].words[0].w == "hello"
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 1.2


def test_transcribe_uses_turbo_repo_when_requested(monkeypatch, tmp_path):
    fixture = {"language": "en", "segments": []}
    capture = {}
    _install_fake_mlx_whisper(monkeypatch, fixture, capture)

    result = transcribe(tmp_path / "a.wav", turbo=True, log=LOG)

    assert capture["repo"] == TURBO_REPO
    assert capture["word_timestamps"] is True
    assert result.model == "large-v3-turbo"
    assert result.turbo is True


def test_transcribe_uses_full_repo_by_default(monkeypatch, tmp_path):
    fixture = {"language": "ru", "segments": []}
    capture = {}
    _install_fake_mlx_whisper(monkeypatch, fixture, capture)

    transcribe(tmp_path / "a.wav", turbo=False, log=LOG)

    assert capture["repo"] == FULL_REPO


def test_transcribe_passes_language_and_antihallucination(monkeypatch, tmp_path):
    fixture = {"language": "ru", "segments": []}
    capture = {}
    _install_fake_mlx_whisper(monkeypatch, fixture, capture)

    transcribe(tmp_path / "a.wav", turbo=False, log=LOG, language="ru", initial_prompt="glossary here")

    assert capture["language"] == "ru"
    assert capture["condition_on_previous_text"] is False
    assert capture["hallucination_silence_threshold"] == 2.0
    assert capture["initial_prompt"] == "glossary here"
    assert capture["word_timestamps"] is True


def test_transcribe_omits_language_when_auto(monkeypatch, tmp_path):
    fixture = {"language": "ru", "segments": []}
    capture = {}
    _install_fake_mlx_whisper(monkeypatch, fixture, capture)

    transcribe(tmp_path / "a.wav", turbo=False, log=LOG, language=None, initial_prompt=None)

    assert "language" not in capture


def test_build_initial_prompt_appends_extra():
    assert build_initial_prompt("") == DEFAULT_INITIAL_PROMPT
    combined = build_initial_prompt("ФизТех, Богодаров")
    assert combined.startswith(DEFAULT_INITIAL_PROMPT)
    assert "ФизТех, Богодаров" in combined


def _seg(text):
    from transcriber.models import AsrSegment
    return AsrSegment(start=0.0, end=1.0, text=text, words=[])


def test_filter_artifact_segments_drops_known_hallucinations():
    from transcriber.stages.asr_mlx import filter_artifact_segments

    segs = [
        _seg("Продолжение следует..."),
        _seg("Спасибо за просмотр!"),
        _seg("Thank you for watching! Thank you for watching!"),
        _seg("Субтитры сделал DimaTorzok"),
        _seg("Редактор субтитров А.Синецкая"),
        _seg("Да, про хакатон говорили."),
        _seg("Спасибо, я понял."),
    ]
    kept = filter_artifact_segments(segs)
    texts = [s.text for s in kept]
    assert texts == ["Да, про хакатон говорили.", "Спасибо, я понял."]


def test_filter_artifact_segments_honors_extra_denylist():
    from transcriber.stages.asr_mlx import filter_artifact_segments

    segs = [_seg("Реклама казино вулкан"), _seg("нормальный текст")]
    kept = filter_artifact_segments(segs, extra=["реклама казино вулкан"])
    assert [s.text for s in kept] == ["нормальный текст"]
