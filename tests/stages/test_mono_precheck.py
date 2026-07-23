import logging
import subprocess

from transcriber.models import DiarResult, DiarSegment
from transcriber.stages.mono_precheck import (
    SHORT_FULL_SEC,
    decide_mono,
    is_likely_monologue,
)

LOG = logging.getLogger("test")


def _make_silent_wav(path, duration_sec=2, sample_rate=44100):
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"anullsrc=r={sample_rate}:cl=mono",
            "-t", str(duration_sec), str(path),
        ],
        check=True,
        capture_output=True,
    )


def _diar_with_speakers(*speakers: str) -> DiarResult:
    total = {s: 1.0 for s in speakers}
    segments = [DiarSegment(start=0.0, end=1.0, speaker=s) for s in speakers]
    return DiarResult(segments=segments, embeddings={}, total_speech_sec=total)


# --- decide_mono (pure) --------------------------------------------------

def test_decide_mono_all_single_speaker_is_true():
    assert decide_mono([1, 1, 1]) is True


def test_decide_mono_any_multi_speaker_is_false():
    assert decide_mono([1, 2, 1]) is False


def test_decide_mono_empty_is_none():
    assert decide_mono([]) is None


# --- is_likely_monologue --------------------------------------------------

def test_short_file_uses_single_window_covering_whole_file(tmp_path):
    wav = tmp_path / "src.wav"
    _make_silent_wav(wav, duration_sec=2)
    duration = 2.0
    assert duration <= SHORT_FULL_SEC

    calls = []

    def fake_diarize(window_wav, device, num_speakers, min_speakers, max_speakers, log):
        calls.append((window_wav, device, num_speakers, min_speakers, max_speakers))
        return _diar_with_speakers("SPEAKER_00")

    result = is_likely_monologue(wav, duration, fake_diarize, "cpu", LOG)

    assert result is True
    assert len(calls) == 1
    _, device, num_speakers, min_speakers, max_speakers = calls[0]
    assert device == "cpu"
    assert (num_speakers, min_speakers, max_speakers) == (None, None, None)


def test_long_file_samples_multiple_windows(tmp_path):
    wav = tmp_path / "src.wav"
    _make_silent_wav(wav, duration_sec=2)  # real file is short; declared duration is not
    duration = 600.0  # 10 min -> long-file path, multiple 30s windows
    assert duration > SHORT_FULL_SEC

    calls = []

    def fake_diarize(window_wav, device, num_speakers, min_speakers, max_speakers, log):
        calls.append(window_wav)
        return _diar_with_speakers("SPEAKER_00")

    result = is_likely_monologue(wav, duration, fake_diarize, "cpu", LOG)

    assert result is True
    assert len(calls) == 3  # matches langdetect.window_starts(600, 3) window count


def test_returns_false_when_any_window_has_multiple_speakers(tmp_path):
    wav = tmp_path / "src.wav"
    _make_silent_wav(wav, duration_sec=2)
    duration = 600.0

    responses = [
        _diar_with_speakers("SPEAKER_00"),
        _diar_with_speakers("SPEAKER_00", "SPEAKER_01"),
        _diar_with_speakers("SPEAKER_00"),
    ]

    def fake_diarize(window_wav, device, num_speakers, min_speakers, max_speakers, log):
        return responses.pop(0)

    result = is_likely_monologue(wav, duration, fake_diarize, "cpu", LOG)

    assert result is False


def test_returns_none_when_every_window_fails(tmp_path):
    wav = tmp_path / "src.wav"
    _make_silent_wav(wav, duration_sec=2)
    duration = 600.0

    def fake_diarize(window_wav, device, num_speakers, min_speakers, max_speakers, log):
        raise RuntimeError("boom")

    result = is_likely_monologue(wav, duration, fake_diarize, "cpu", LOG)

    assert result is None


def test_partial_window_failure_still_decides_from_successful_windows(tmp_path):
    wav = tmp_path / "src.wav"
    _make_silent_wav(wav, duration_sec=2)
    duration = 600.0

    responses = [RuntimeError("boom"), _diar_with_speakers("SPEAKER_00"), _diar_with_speakers("SPEAKER_00")]

    def fake_diarize(window_wav, device, num_speakers, min_speakers, max_speakers, log):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    result = is_likely_monologue(wav, duration, fake_diarize, "cpu", LOG)

    assert result is True


def test_speaker_count_falls_back_to_unique_segment_speakers_when_total_speech_empty(tmp_path):
    wav = tmp_path / "src.wav"
    _make_silent_wav(wav, duration_sec=2)
    duration = 2.0

    def fake_diarize(window_wav, device, num_speakers, min_speakers, max_speakers, log):
        segments = [
            DiarSegment(start=0.0, end=1.0, speaker="SPEAKER_00"),
            DiarSegment(start=1.0, end=2.0, speaker="SPEAKER_01"),
        ]
        return DiarResult(segments=segments, embeddings={}, total_speech_sec={})

    result = is_likely_monologue(wav, duration, fake_diarize, "cpu", LOG)

    assert result is False
