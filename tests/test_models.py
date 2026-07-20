from transcriber.models import (
    AsrInfo,
    KeyTopic,
    RawDoc,
    Segment,
    SpeakerMeta,
    Summary,
    TopicRef,
    Word,
)


def _sample_doc() -> RawDoc:
    return RawDoc(
        schema=1,
        content_hash="blake2b:abc123",
        source_name="team call.m4a",
        source_path="/abs/team call.m4a",
        language="ru",
        duration_sec=754.2,
        num_speakers=2,
        is_monologue=False,
        asr=AsrInfo(backend="mlx", model="large-v3", turbo=False),
        created_at="2026-07-19T21:00:00Z",
        segments=[
            Segment(
                start=0.0,
                end=3.1,
                speaker="SPEAKER_00",
                text="Hi, how are you",
                words=[Word(w="Hi,", start=0.0, end=0.4, speaker="SPEAKER_00")],
            )
        ],
        speakers_meta=[
            SpeakerMeta(label="SPEAKER_00", name=None, embedding=[0.1, 0.2], total_speech_sec=512.3)
        ],
        summary=Summary(
            title="Call with Jamie",
            text="Summary text",
            topics=[TopicRef(term="budget", ts=12.0)],
            hashtags=["budget", "cyprus"],
            key_topics=[KeyTopic(topic="Move to Cyprus", ts=225.0, note="gist")],
            decisions=[],
            length_tier="5-8",
            generated=True,
            llm_model="qwen2.5:14b",
        ),
    )


def test_rawdoc_roundtrip_through_dict():
    doc = _sample_doc()
    restored = RawDoc.from_dict(doc.to_dict())
    assert restored == doc


def test_rawdoc_roundtrip_without_summary():
    doc = _sample_doc()
    doc.summary = None
    restored = RawDoc.from_dict(doc.to_dict())
    assert restored.summary is None
    assert restored == doc


def test_rawdoc_to_dict_is_json_serializable():
    import json

    doc = _sample_doc()
    json.dumps(doc.to_dict(), ensure_ascii=False)
