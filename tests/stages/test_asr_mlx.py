import logging
import sys
import types

from transcriber.stages.asr_mlx import FULL_REPO, TURBO_REPO, transcribe

LOG = logging.getLogger("test")


def _install_fake_mlx_whisper(monkeypatch, fixture, capture=None):
    fake_module = types.ModuleType("mlx_whisper")

    def fake_transcribe(path, path_or_hf_repo, word_timestamps):
        if capture is not None:
            capture["path"] = path
            capture["repo"] = path_or_hf_repo
            capture["word_timestamps"] = word_timestamps
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
