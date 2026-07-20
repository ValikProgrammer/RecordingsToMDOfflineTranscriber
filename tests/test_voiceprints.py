from transcriber.models import RawDoc, Segment, SpeakerMeta, Word
from transcriber.voiceprints import VoiceprintStore, cosine_similarity, identify_speakers


def test_store_creates_directory(tmp_path):
    store_dir = tmp_path / "voiceprints"
    VoiceprintStore(store_dir)
    assert store_dir.exists()


def test_cosine_similarity_basics():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([], [1.0]) == 0.0


def test_enroll_then_identify_matches(tmp_path):
    store = VoiceprintStore(tmp_path / "voiceprints")
    store.enroll("Jamie", [1.0, 0.0, 0.0])
    assert store.identify([0.9, 0.1, 0.0], threshold=0.7) == "Jamie"


def test_identify_returns_none_below_threshold(tmp_path):
    store = VoiceprintStore(tmp_path / "voiceprints")
    store.enroll("Jamie", [1.0, 0.0, 0.0])
    assert store.identify([0.0, 1.0, 0.0], threshold=0.7) is None


def test_identify_picks_best_of_several(tmp_path):
    store = VoiceprintStore(tmp_path / "voiceprints")
    store.enroll("Jamie", [1.0, 0.0, 0.0])
    store.enroll("Alex", [0.0, 1.0, 0.0])
    assert store.identify([0.1, 0.95, 0.0], threshold=0.7) == "Alex"


def test_enroll_persists_across_instances(tmp_path):
    store_dir = tmp_path / "voiceprints"
    VoiceprintStore(store_dir).enroll("Jamie", [1.0, 0.0])
    assert VoiceprintStore(store_dir).identify([1.0, 0.0], threshold=0.7) == "Jamie"


def _doc_with_speaker(label, embedding, name=None):
    seg = Segment(0.0, 1.0, label, "hi", [Word("hi", 0.0, 1.0, label)])
    return RawDoc(
        schema=1, content_hash="h", source_name="a.m4a", source_path="/a.m4a",
        language="ru", duration_sec=1.0, num_speakers=1, is_monologue=False,
        asr=None, created_at="", segments=[seg],
        speakers_meta=[SpeakerMeta(label=label, name=name, embedding=embedding, total_speech_sec=1.0)],
        summary=None,
    )


def test_identify_speakers_renames_matched_unnamed_speaker(tmp_path):
    store = VoiceprintStore(tmp_path / "voiceprints")
    store.enroll("Галя", [1.0, 0.0, 0.0])
    doc = _doc_with_speaker("SPEAKER_00", [0.98, 0.02, 0.0])

    identify_speakers(doc, store, threshold=0.7)

    assert doc.segments[0].speaker == "Галя"
    assert doc.segments[0].words[0].speaker == "Галя"
    assert doc.speakers_meta[0].name == "Галя"


def test_identify_speakers_leaves_unmatched_speaker(tmp_path):
    store = VoiceprintStore(tmp_path / "voiceprints")
    store.enroll("Галя", [1.0, 0.0, 0.0])
    doc = _doc_with_speaker("SPEAKER_00", [0.0, 1.0, 0.0])

    identify_speakers(doc, store, threshold=0.7)

    assert doc.segments[0].speaker == "SPEAKER_00"
    assert doc.speakers_meta[0].name is None


def test_identify_speakers_does_not_touch_already_named(tmp_path):
    store = VoiceprintStore(tmp_path / "voiceprints")
    store.enroll("Галя", [1.0, 0.0, 0.0])
    doc = _doc_with_speaker("SPEAKER_00", [1.0, 0.0, 0.0], name="Иван")

    identify_speakers(doc, store, threshold=0.7)

    assert doc.speakers_meta[0].name == "Иван"
