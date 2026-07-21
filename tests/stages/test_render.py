from transcriber.models import (
    AsrInfo,
    Decision,
    KeyTopic,
    RawDoc,
    Segment,
    Summary,
    TopicRef,
)
from transcriber.stages.render import format_timecode, render_markdown, yaml_escape


def _base_doc(**overrides) -> RawDoc:
    defaults = dict(
        schema=1,
        content_hash="blake2b:x",
        source_name="team call.m4a",
        source_path="/a.m4a",
        language="ru",
        duration_sec=754.0,
        num_speakers=2,
        is_monologue=False,
        asr=AsrInfo(backend="mlx", model="large-v3", turbo=False),
        created_at="2026-07-19T21:00:00Z",
        segments=[
            Segment(0.0, 3.0, "SPEAKER_00", "Hi, how are you"),
            Segment(15.0, 18.0, "SPEAKER_01", "Good, and you"),
        ],
        speakers_meta=[],
        summary=Summary(
            title="Call with Jamie",
            text="Short summary of the conversation.",
            topics=[TopicRef(term="budget", ts=12.0), TopicRef(term="Cyprus", ts=225.0)],
            hashtags=["budget", "cyprus", "2026"],
            generated=True,
            llm_model="qwen2.5:14b",
        ),
    )
    defaults.update(overrides)
    return RawDoc(**defaults)


def test_format_timecode_mmss_and_hhmmss():
    assert format_timecode(75, use_hours=False) == "01:15"
    assert format_timecode(3675, use_hours=True) == "01:01:15"


def test_yaml_escape_doubles_inner_quotes():
    assert yaml_escape('He said "hi"') == '"He said \\"hi\\""'


def test_render_dialogue_includes_frontmatter_and_speaker_labels():
    doc = _base_doc()
    md = render_markdown(doc, "2026-07-12", "Call with Jamie")
    assert md.startswith("---\n")
    assert 'Title: "Call with Jamie"' in md
    assert "Date: 2026-07-12" in md
    assert "Language: RU" in md
    assert "Speakers: 2" in md
    assert 'Duration: "12:34"' in md
    assert 'Source file: "team call.m4a"' in md
    assert "tags: [budget, cyprus]" in md  # numeric "2026" hashtag dropped; key stays lowercase for Obsidian
    # body must NOT duplicate frontmatter metadata when frontmatter is on
    assert "**Date:**" not in md
    assert "**Source file:**" not in md
    assert "**[00:00] Speaker 1:** Hi, how are you" in md
    assert "**[00:15] Speaker 2:** Good, and you" in md
    assert "**Topics:**\n- [00:12] budget\n- [03:45] Cyprus" in md
    assert "**Hashtags:** #budget #cyprus #2026" in md


def test_transcript_override_replaces_verbatim_segments():
    doc = _base_doc()
    md = render_markdown(
        doc, "2026-07-12", "Call with Jamie",
        transcript_override="[00:00] Причёсанный текст блока.",
    )
    # keeps the frontmatter + summary scaffolding
    assert md.startswith("---\n")
    assert "### Summary" in md
    assert "### Transcript" in md
    # the override body appears; the verbatim per-segment lines do not
    assert "[00:00] Причёсанный текст блока." in md
    assert "**[00:00] Speaker 1:** Hi, how are you" not in md


def test_topics_render_one_per_line_timecode_first():
    topics = [TopicRef(term=f"Topic {i}", ts=float(i * 60)) for i in range(30)]
    doc = _base_doc()
    doc.summary.topics = topics
    md = render_markdown(doc, "2026-07-12", "Long call")

    assert "**Topics:**" in md
    assert "- [00:00] Topic 0" in md
    assert "- [29:00] Topic 29" in md
    assert md.count("\n- [") >= 30
    assert " · Topic" not in md


def test_render_long_form_adds_key_topics_and_decisions_sections():
    doc = _base_doc(duration_sec=3000.0)  # 50 min >= 45 min long_form_from_min
    doc.summary.key_topics = [KeyTopic(topic="Move to Cyprus", ts=225.0, note="gist")]
    doc.summary.decisions = [Decision(text="Decided not to sell the apartment", ts=2470.0)]
    md = render_markdown(doc, "2026-07-12", "Call with Jamie", long_form_from_min=45)
    assert "### Key Topics" in md
    assert "- **Move to Cyprus** (03:45) — gist" in md
    assert "### Decisions" in md
    assert "- Decided not to sell the apartment (41:10)" in md


def test_render_short_form_omits_long_form_sections_even_if_present():
    doc = _base_doc(duration_sec=300.0)
    doc.summary.key_topics = [KeyTopic(topic="X", ts=1.0, note="y")]
    doc.summary.decisions = [Decision(text="Z", ts=1.0)]
    md = render_markdown(doc, "2026-07-12", "Call with Jamie", long_form_from_min=45)
    assert "### Key Topics" not in md
    assert "### Decisions" not in md


def test_render_monologue_omits_speaker_labels():
    doc = _base_doc(
        num_speakers=1,
        is_monologue=True,
        segments=[Segment(0.0, 3.0, None, "Just talking to myself")],
    )
    md = render_markdown(doc, "2026-07-12", "Note")
    assert "**[00:00]** Just talking to myself" in md
    assert "Speaker 1" not in md


def test_render_monologue_with_name_shows_name():
    doc = _base_doc(
        num_speakers=1,
        is_monologue=True,
        segments=[Segment(0.0, 3.0, "Alex", "Voice memo note")],
    )
    md = render_markdown(doc, "2026-07-12", "Note")
    assert "**[00:00] Alex:** Voice memo note" in md


def test_render_wikilink_speakers_only_applies_to_named():
    doc = _base_doc(segments=[Segment(0.0, 3.0, "Jamie", "Hi")])
    md = render_markdown(doc, "2026-07-12", "T", wikilink_speakers=True)
    assert "[[Jamie]]" in md

    doc2 = _base_doc(segments=[Segment(0.0, 3.0, "SPEAKER_00", "Hi")])
    md2 = render_markdown(doc2, "2026-07-12", "T", wikilink_speakers=True)
    assert "[[Speaker 1]]" not in md2
    assert "Speaker 1" in md2


def test_render_no_frontmatter_flag():
    doc = _base_doc()
    md = render_markdown(doc, "2026-07-12", "T", frontmatter=False)
    assert not md.startswith("---")
    assert "Title:" not in md
    # with frontmatter off, metadata falls back to the body block
    assert "**Duration:**" in md
    assert "**Source file:**" in md


def test_render_text_mode_no_summary_omits_summary_block_and_tags():
    doc = _base_doc(summary=None)
    md = render_markdown(doc, "2026-07-12", "T")
    assert "### Summary" not in md
    assert "tags:" not in md
    assert "### Transcript" in md


def test_render_hour_plus_duration_uses_hhmmss_timecodes():
    doc = _base_doc(duration_sec=3700.0, segments=[Segment(3650.0, 3660.0, "SPEAKER_00", "text")])
    md = render_markdown(doc, "2026-07-12", "T")
    assert 'Duration: "01:01:40"' in md
    assert "**[01:00:50]" in md


def test_hashtags_with_leading_hash_are_normalized_to_single():
    doc = _base_doc()
    doc.summary.hashtags = ["#хакатоны", "##расписание", "субботняя"]
    md = render_markdown(doc, "2026-07-12", "T")
    assert "**Hashtags:** #хакатоны #расписание #субботняя" in md
    assert "##" not in md.split("**Hashtags:**")[1].split("\n")[0]
