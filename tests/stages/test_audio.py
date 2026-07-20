import json
import subprocess

import pytest

from transcriber.stages.audio import FfmpegNotFoundError, check_ffmpeg_available, normalize


def _make_silent_wav(path, duration_sec=1, sample_rate=44100, channels=2):
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"anullsrc=r={sample_rate}:cl={'stereo' if channels == 2 else 'mono'}",
            "-t", str(duration_sec), str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_check_ffmpeg_available_does_not_raise_when_installed():
    check_ffmpeg_available()


def test_check_ffmpeg_available_raises_when_missing(monkeypatch):
    monkeypatch.setattr("transcriber.stages.audio.shutil.which", lambda name: None)
    with pytest.raises(FfmpegNotFoundError):
        check_ffmpeg_available()


def test_normalize_produces_16k_mono_wav_with_correct_duration(tmp_path):
    src = tmp_path / "src.wav"
    _make_silent_wav(src, duration_sec=2, sample_rate=44100, channels=2)

    out_path, duration = normalize(src, tmp_path / "tmp")

    assert out_path.exists()
    assert 1.9 <= duration <= 2.1

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "json", str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(probe.stdout)["streams"][0]
    assert info["sample_rate"] == "16000"
    assert info["channels"] == 1
