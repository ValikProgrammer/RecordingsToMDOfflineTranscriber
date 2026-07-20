import logging

import pytest

from transcriber.stages.diarize import (
    MissingHfTokenError,
    build_diar_result,
    diarize,
    extract_embeddings,
)

LOG = logging.getLogger("test")


class FakeTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeAnnotation:
    def __init__(self, tracks, labels):
        self._tracks = tracks
        self._labels = labels

    def itertracks(self, yield_label=True):
        for turn, track_id, speaker in self._tracks:
            yield turn, track_id, speaker

    def labels(self):
        return self._labels

    def label_timeline(self, speaker):
        return f"timeline-{speaker}"


class FakeOutput:
    """Mimics pyannote.audio 4.x DiarizeOutput."""

    def __init__(self, annotation, speaker_embeddings=None):
        self.speaker_diarization = annotation
        self.speaker_embeddings = speaker_embeddings


def test_diarize_raises_without_hf_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(MissingHfTokenError):
        diarize(tmp_path / "a.wav", "mps", None, None, None, LOG)


def test_build_diar_result_aggregates_speech_time_per_speaker():
    tracks = [
        (FakeTurn(0.0, 2.0), "t1", "SPEAKER_00"),
        (FakeTurn(2.0, 3.5), "t1", "SPEAKER_01"),
        (FakeTurn(3.5, 5.0), "t2", "SPEAKER_00"),
    ]
    annotation = FakeAnnotation(tracks, ["SPEAKER_00", "SPEAKER_01"])
    result = build_diar_result(FakeOutput(annotation), LOG)

    assert result.total_speech_sec["SPEAKER_00"] == 3.5
    assert result.total_speech_sec["SPEAKER_01"] == 1.5
    assert len(result.segments) == 3
    assert result.segments[0].speaker == "SPEAKER_00"


def test_build_diar_result_accepts_raw_annotation():
    # legacy mode / older pyannote returns the Annotation directly, not a DiarizeOutput
    annotation = FakeAnnotation([(FakeTurn(0.0, 1.0), "t1", "SPEAKER_00")], ["SPEAKER_00"])
    result = build_diar_result(annotation, LOG)

    assert result.total_speech_sec["SPEAKER_00"] == 1.0
    assert result.embeddings == {}


def test_extract_embeddings_missing_embeddings_returns_empty():
    annotation = FakeAnnotation([], ["SPEAKER_00"])
    embeddings = extract_embeddings(FakeOutput(annotation), annotation, LOG)
    assert embeddings == {}


def test_extract_embeddings_returns_vector_per_speaker():
    annotation = FakeAnnotation([], ["SPEAKER_00", "SPEAKER_01"])
    output = FakeOutput(annotation, speaker_embeddings=[[0.1, 0.2], [0.3, 0.4]])
    embeddings = extract_embeddings(output, annotation, LOG)
    assert embeddings == {"SPEAKER_00": [0.1, 0.2], "SPEAKER_01": [0.3, 0.4]}


def test_extract_embeddings_skips_speaker_when_row_missing():
    annotation = FakeAnnotation([], ["SPEAKER_00", "SPEAKER_01"])
    # only one embedding row for two speakers -> second lookup fails and is skipped
    output = FakeOutput(annotation, speaker_embeddings=[[0.1, 0.2]])
    embeddings = extract_embeddings(output, annotation, LOG)
    assert embeddings == {"SPEAKER_00": [0.1, 0.2]}
