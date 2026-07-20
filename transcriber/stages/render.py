"""RawDoc -> Markdown per the Obsidian-ready template (§11)."""
from __future__ import annotations

from ..models import RawDoc, Segment
from ..naming import slugify_tag


def format_timecode(seconds: float, use_hours: bool) -> str:
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if use_hours else f"{m:02d}:{s:02d}"


def yaml_escape(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def speaker_order_of(segments: list[Segment]) -> list[str]:
    order: list[str] = []
    for seg in segments:
        if seg.speaker and seg.speaker.startswith("SPEAKER_") and seg.speaker not in order:
            order.append(seg.speaker)
    return order


def speaker_display(speaker: str | None, order: list[str], wikilink: bool) -> str | None:
    if speaker is None:
        return None
    if speaker.startswith("SPEAKER_"):
        idx = order.index(speaker) + 1
        return f"Speaker {idx}"
    return f"[[{speaker}]]" if wikilink else speaker


def _render_frontmatter(title: str, doc: RawDoc, day_str: str, duration_str: str, tags: list[str]) -> str:
    lines = ["---"]
    lines.append(f"title: {yaml_escape(title)}")
    lines.append(f"date: {day_str}")
    lines.append(f"language: {doc.language.upper()}")
    lines.append(f"speakers: {doc.num_speakers}")
    lines.append(f"duration: {yaml_escape(duration_str)}")
    lines.append(f"source_file: {yaml_escape(doc.source_name)}")
    if tags:
        lines.append(f"tags: [{', '.join(tags)}]")
    lines.append("---")
    return "\n".join(lines)


def render_markdown(
    doc: RawDoc,
    day_str: str,
    title: str,
    *,
    frontmatter: bool = True,
    wikilink_speakers: bool = False,
    long_form_from_min: float = 45,
) -> str:
    use_hours = doc.duration_sec >= 3600
    duration_str = format_timecode(doc.duration_sec, use_hours)
    order = speaker_order_of(doc.segments)
    tags = [t for t in (slugify_tag(h) for h in (doc.summary.hashtags if doc.summary else [])) if t]

    parts: list[str] = []
    if frontmatter:
        parts.append(_render_frontmatter(title, doc, day_str, duration_str, tags))
        parts.append("")

    parts.append(f"# {title}")
    parts.append("")
    parts.append(
        f"**Date:** {day_str}  ·  **Language:** {doc.language.upper()}  ·  "
        f"**Speakers:** {doc.num_speakers}  ·  **Duration:** {duration_str}"
    )
    parts.append(f"**Source file:** `{doc.source_name}`")
    parts.append("")

    if doc.summary:
        parts.append("### Summary")
        parts.append(doc.summary.text)
        parts.append("")

        if doc.summary.topics:
            parts.append("**Topics:**")
            for t in doc.summary.topics:
                parts.append(f"- [{format_timecode(t.ts, use_hours)}] {t.term}")
            parts.append("")

        if doc.summary.hashtags:
            hashtags_str = " ".join(f"#{h}" for h in doc.summary.hashtags)
            parts.append(f"**Hashtags:** {hashtags_str}")
            parts.append("")

        if (doc.duration_sec / 60) >= long_form_from_min:
            if doc.summary.key_topics:
                parts.append("### Key Topics")
                for kt in doc.summary.key_topics:
                    parts.append(f"- **{kt.topic}** ({format_timecode(kt.ts, use_hours)}) — {kt.note}")
                parts.append("")
            if doc.summary.decisions:
                parts.append("### Decisions")
                for d in doc.summary.decisions:
                    parts.append(f"- {d.text} ({format_timecode(d.ts, use_hours)})")
                parts.append("")

    parts.append("---")
    parts.append("")
    parts.append("### Transcript")
    for seg in doc.segments:
        ts = format_timecode(seg.start, use_hours)
        label = speaker_display(seg.speaker, order, wikilink_speakers)
        if label is None:
            parts.append(f"**[{ts}]** {seg.text}")
        else:
            parts.append(f"**[{ts}] {label}:** {seg.text}")

    return "\n".join(parts) + "\n"
