import logging

from transcriber.models import AsrResult, AsrSegment, AsrWord, DiarResult, DiarSegment, Segment, Word
from transcriber.stages.merge import (
    apply_names,
    assign_speaker,
    build_segments,
    build_text_doc,
    collapse_monologue,
    compute_total_speech,
    fold_phantom_speakers,
    merge,
)

LOG = logging.getLogger("test")


def _diar(*segs):
    return DiarResult(segments=[DiarSegment(*s) for s in segs])


def test_assign_speaker_picks_max_overlap():
    diar = _diar((0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01"))
    assert assign_speaker(1.5, 2.5, diar) == "SPEAKER_00"  # 0.5s overlap vs 0.5s -> first wins on >
    assert assign_speaker(1.0, 3.9, diar) == "SPEAKER_01"  # 1.9s overlap vs 1.0s


def test_assign_speaker_picks_nearest_when_no_overlap():
    diar = _diar((0.0, 1.0, "SPEAKER_00"), (5.0, 6.0, "SPEAKER_01"))
    assert assign_speaker(1.2, 1.4, diar) == "SPEAKER_00"
    assert assign_speaker(4.5, 4.8, diar) == "SPEAKER_01"


def test_assign_speaker_no_diar_segments_returns_none():
    assert assign_speaker(0.0, 1.0, DiarResult()) is None


def _asr(*words):
    return AsrResult(
        language="ru",
        segments=[AsrSegment(start=words[0][1], end=words[-1][2], text="", words=[AsrWord(*w) for w in words])],
        backend="mlx",
        model="large-v3",
        turbo=False,
    )


def test_build_segments_splits_on_speaker_change():
    asr = _asr(("Hi", 0.0, 0.5), ("yes", 0.6, 1.0))
    diar = _diar((0.0, 0.5, "SPEAKER_00"), (0.6, 1.0, "SPEAKER_01"))
    segments = build_segments(asr, diar)
    assert [s.speaker for s in segments] == ["SPEAKER_00", "SPEAKER_01"]


def test_build_segments_splits_on_long_pause():
    asr = _asr(("Hi", 0.0, 0.5), ("yes", 3.0, 3.5))
    diar = _diar((0.0, 4.0, "SPEAKER_00"))
    segments = build_segments(asr, diar)
    assert len(segments) == 2


def test_build_segments_keeps_short_pause_together():
    asr = _asr(("Hi", 0.0, 0.5), ("yes", 1.2, 1.6))
    diar = _diar((0.0, 2.0, "SPEAKER_00"))
    segments = build_segments(asr, diar)
    assert len(segments) == 1
    assert segments[0].text == "Hi yes"


def test_compute_total_speech_ignores_none_speaker():
    from transcriber.models import Segment

    segments = [
        Segment(0.0, 2.0, "SPEAKER_00", "a"),
        Segment(2.0, 3.0, None, "b"),
    ]
    assert compute_total_speech(segments) == {"SPEAKER_00": 2.0}


def test_collapse_monologue_single_speaker():
    from transcriber.models import Segment

    segments = [Segment(0.0, 2.0, "SPEAKER_00", "a")]
    assert collapse_monologue(segments, 0.92) is True


def test_collapse_monologue_dominant_share_over_threshold():
    from transcriber.models import Segment

    segments = [
        Segment(0.0, 95.0, "SPEAKER_00", "a"),
        Segment(95.0, 100.0, "SPEAKER_01", "b"),
    ]
    assert collapse_monologue(segments, 0.92) is True


def test_collapse_monologue_keeps_dialogue_when_balanced():
    from transcriber.models import Segment

    segments = [
        Segment(0.0, 50.0, "SPEAKER_00", "a"),
        Segment(50.0, 100.0, "SPEAKER_01", "b"),
    ]
    assert collapse_monologue(segments, 0.92) is False


def test_collapse_monologue_no_speakers_is_false():
    from transcriber.models import Segment

    assert collapse_monologue([Segment(0.0, 1.0, None, "a")], 0.92) is False


def test_apply_names_maps_by_first_appearance_order():
    from transcriber.models import Segment

    segments = [
        Segment(0.0, 1.0, "SPEAKER_01", "b"),  # SPEAKER_01 speaks first
        Segment(1.0, 2.0, "SPEAKER_00", "a"),
    ]
    name_map = apply_names(segments, ["Alex", "Jamie"])
    assert name_map == {"SPEAKER_01": "Alex", "SPEAKER_00": "Jamie"}


def test_apply_names_leftover_speaker_uncovered():
    from transcriber.models import Segment

    segments = [
        Segment(0.0, 1.0, "SPEAKER_00", "a"),
        Segment(1.0, 2.0, "SPEAKER_01", "b"),
    ]
    name_map = apply_names(segments, ["Alex"])
    assert name_map == {"SPEAKER_00": "Alex"}
    assert "SPEAKER_01" not in name_map


def test_apply_names_no_names_returns_empty():
    from transcriber.models import Segment

    assert apply_names([Segment(0.0, 1.0, "SPEAKER_00", "a")], None) == {}


def test_merge_end_to_end_dialogue_produces_rawdoc_with_speakers_meta():
    asr = _asr(("Hi", 0.0, 0.5), ("yes", 3.0, 3.5))
    diar = DiarResult(
        segments=[DiarSegment(0.0, 0.5, "SPEAKER_00"), DiarSegment(3.0, 3.5, "SPEAKER_01")],
        embeddings={"SPEAKER_00": [0.1, 0.2], "SPEAKER_01": [0.3, 0.4]},
    )
    doc = merge(
        asr, diar, mono_threshold=0.92, names=None, log=LOG,
        content_hash="blake2b:x", source_name="a.m4a", source_path="/a.m4a", duration_sec=4.0,
    )
    assert doc.is_monologue is False
    assert doc.num_speakers == 2
    labels = {sm.label: sm.embedding for sm in doc.speakers_meta}
    assert labels["SPEAKER_00"] == [0.1, 0.2]
    assert labels["SPEAKER_01"] == [0.3, 0.4]


def test_merge_monologue_strips_unnamed_labels():
    asr = _asr(("Hi", 0.0, 0.5))
    diar = DiarResult(segments=[DiarSegment(0.0, 0.5, "SPEAKER_00")])
    doc = merge(
        asr, diar, mono_threshold=0.92, names=None, log=LOG,
        content_hash="blake2b:x", source_name="a.m4a", source_path="/a.m4a", duration_sec=0.5,
    )
    assert doc.is_monologue is True
    assert doc.segments[0].speaker is None


def test_merge_monologue_with_name_keeps_name():
    asr = _asr(("Hi", 0.0, 0.5))
    diar = DiarResult(segments=[DiarSegment(0.0, 0.5, "SPEAKER_00")])
    doc = merge(
        asr, diar, mono_threshold=0.92, names=["Alex"], log=LOG,
        content_hash="blake2b:x", source_name="a.m4a", source_path="/a.m4a", duration_sec=0.5,
    )
    assert doc.is_monologue is True
    assert doc.segments[0].speaker == "Alex"


def test_build_text_doc_has_no_speakers():
    asr = _asr(("Hi", 0.0, 0.5), ("world", 0.6, 1.0))
    doc = build_text_doc(
        asr, content_hash="blake2b:x", source_name="a.m4a", source_path="/a.m4a", duration_sec=1.0
    )
    assert doc.num_speakers == 0
    assert doc.is_monologue is True
    assert all(s.speaker is None for s in doc.segments)
    assert doc.speakers_meta == []


def test_fold_phantom_speakers_merges_tiny_speaker_into_nearest():
    segs = [
        Segment(0.0, 100.0, "SPEAKER_00", "a", [Word("a", 0.0, 100.0, "SPEAKER_00")]),
        Segment(100.0, 190.0, "SPEAKER_01", "b", [Word("b", 100.0, 190.0, "SPEAKER_01")]),
        Segment(190.0, 191.0, "SPEAKER_02", "c", [Word("c", 190.0, 191.0, "SPEAKER_02")]),
    ]
    out = fold_phantom_speakers(segs, min_share=0.02)
    speakers = {s.speaker for s in out}
    assert speakers == {"SPEAKER_00", "SPEAKER_01"}
    assert out[2].speaker == "SPEAKER_01"  # nearest by time
    assert out[2].words[0].speaker == "SPEAKER_01"


def test_fold_phantom_speakers_noop_when_threshold_zero():
    segs = [
        Segment(0.0, 100.0, "SPEAKER_00", "a", []),
        Segment(190.0, 191.0, "SPEAKER_02", "c", []),
    ]
    out = fold_phantom_speakers(segs, min_share=0.0)
    assert {s.speaker for s in out} == {"SPEAKER_00", "SPEAKER_02"}


def test_merge_folds_phantom_speaker_into_two(caplog):
    asr = AsrResult(
        language="ru", backend="mlx", model="large-v3", turbo=False,
        segments=[
            AsrSegment(0.0, 100.0, "aaa", words=[AsrWord("aaa", 0.0, 100.0)]),
            AsrSegment(100.0, 190.0, "bbb", words=[AsrWord("bbb", 100.0, 190.0)]),
            AsrSegment(190.0, 191.0, "ccc", words=[AsrWord("ccc", 190.0, 191.0)]),
        ],
    )
    diar = DiarResult(segments=[
        DiarSegment(0.0, 100.0, "SPEAKER_00"),
        DiarSegment(100.0, 190.0, "SPEAKER_01"),
        DiarSegment(190.0, 191.0, "SPEAKER_02"),
    ])
    doc = merge(asr, diar, mono_threshold=0.99, names=None, log=LOG, min_speaker_share=0.02,
                content_hash="h", source_name="x.m4a", source_path="/x.m4a", duration_sec=191.0)
    assert doc.num_speakers == 2
