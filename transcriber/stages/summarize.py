"""Ollama-backed summary/title/hashtags/topics generation (§10).

ollama is imported lazily inside call_ollama_json() so this module can be
unit-tested (with a fake module) without a running Ollama daemon.
"""
from __future__ import annotations

import json
import logging

from ..config import Config, SummaryTier
from ..models import Decision, KeyTopic, RawDoc, Summary, TopicRef

SYSTEM_PROMPT_TEMPLATE = """You are a bilingual (RU/EN) meeting-note assistant.
Respond ONLY with a single JSON object, no prose, matching this schema:
{{"title": str, "summary": str, "topics": [{{"term": str, "ts_hint": "MM:SS"}}],
"hashtags": [str], "key_topics": [{{"topic": str, "ts_hint": "MM:SS", "note": str}}],
"decisions": [{{"text": str, "ts_hint": "MM:SS"}}]}}
Write summary, topics, hashtags, key_topics, decisions in language code: {language}.
Write "summary" in a neutral, impersonal voice. Name the subjects rather than the people:
prefer passive/topic phrasing
over active phrasing with a vague actor. Never invent a
speaker's name or refer to "the speakers"/"собеседники" as the grammatical subject.
Sentence-count guidance for "summary" (a guideline, NOT a hard rule — the number of
distinct substantive topics should drive the actual length, don't pad to fit): {sentences}.
Extract one "topics" entry per distinct subject discussed, in chronological order.
Do not limit the number of topics — a long recording with many subjects should have many topics. Do not pad with trivial or duplicate topics.
{long_form_hint}
"""

LONG_FORM_HINT = (
    "This is a long recording: also fill key_topics and decisions with real content. "
    "If there are no explicit decisions, return an empty list for decisions."
)
SHORT_FORM_HINT = "This is a short recording: return empty lists for key_topics and decisions."


def select_tier(duration_sec: float, tiers: list[SummaryTier]) -> SummaryTier:
    duration_min = duration_sec / 60
    for tier in tiers:
        if duration_min <= tier.up_to_min:
            return tier
    return tiers[-1]


def format_transcript_with_timestamps(doc: RawDoc) -> list[str]:
    lines = []
    for seg in doc.segments:
        m, s = divmod(int(seg.start), 60)
        lines.append(f"[{m:02d}:{s:02d}] {seg.text}")
    return lines


def chunk_transcript(lines: list[str], max_chars: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        if current and current_len + len(line) > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append(current)
    return chunks


def parse_ts_hint(hint: str, doc: RawDoc) -> float:
    parts = [int(p) for p in hint.split(":")]
    if len(parts) == 2:
        seconds = parts[0] * 60 + parts[1]
    else:
        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
    if not doc.segments:
        return float(seconds)
    nearest = min(doc.segments, key=lambda s: abs(s.start - seconds))
    return nearest.start


def call_ollama_json(model: str, system_prompt: str, user_prompt: str, log: logging.Logger) -> dict:
    import ollama

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error: json.JSONDecodeError | None = None
    for attempt in range(2):
        response = ollama.chat(model=model, format="json", messages=messages)
        content = response["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            last_error = exc
            log.info(f"summary JSON parse failed (attempt {attempt + 1}), retrying")
    raise RuntimeError(f"Ollama returned invalid JSON after a retry: {last_error}")


def summarize(doc: RawDoc, cfg: Config, log: logging.Logger) -> Summary:
    tier = select_tier(doc.duration_sec, cfg.summary_tiers)
    is_long_form = (doc.duration_sec / 60) >= cfg.long_form_from_min
    hint = LONG_FORM_HINT if is_long_form else SHORT_FORM_HINT
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        language=doc.language, sentences=tier.sentences, long_form_hint=hint,
    )

    lines = format_transcript_with_timestamps(doc)
    max_chars = cfg.llm_ctx * 3  # rough chars-per-token budget for the transcript chunk
    chunks = chunk_transcript(lines, max_chars)

    if len(chunks) <= 1:
        raw = call_ollama_json(cfg.llm_model, system_prompt, "\n".join(lines), log)
    else:
        raw = _map_reduce(chunks, cfg, system_prompt, log)

    summary = _to_summary(raw, doc, tier, cfg.llm_model)
    log.info(f"summary done ({cfg.llm_model})")
    return summary


def _map_reduce(chunks: list[list[str]], cfg: Config, system_prompt: str, log: logging.Logger) -> dict:
    partials = []
    for i, chunk in enumerate(chunks):
        log.info(f"summary map chunk {i + 1}/{len(chunks)}")
        partials.append(call_ollama_json(cfg.llm_model, system_prompt, "\n".join(chunk), log))
    reduce_prompt = (
        "Combine these partial summaries (JSON list below) into ONE final JSON object "
        "with the same schema. Keep ALL distinct topics in chronological order; "
        "merge only topics that are duplicates or clearly the same subject. "
        "Do not drop distinct topics or cap their number:\n"
        + json.dumps(partials, ensure_ascii=False)
    )
    return call_ollama_json(cfg.llm_model, system_prompt, reduce_prompt, log)


def _to_summary(raw: dict, doc: RawDoc, tier: SummaryTier, llm_model: str) -> Summary:
    topics = [
        TopicRef(term=t["term"], ts=parse_ts_hint(t["ts_hint"], doc)) for t in raw.get("topics", [])
    ]
    key_topics = [
        KeyTopic(topic=kt["topic"], ts=parse_ts_hint(kt["ts_hint"], doc), note=kt.get("note", ""))
        for kt in raw.get("key_topics", [])
    ]
    decisions = [
        Decision(text=d["text"], ts=parse_ts_hint(d["ts_hint"], doc)) for d in raw.get("decisions", [])
    ]
    return Summary(
        title=raw.get("title", ""),
        text=raw.get("summary", ""),
        topics=topics,
        hashtags=list(raw.get("hashtags", [])),
        key_topics=key_topics,
        decisions=decisions,
        length_tier=tier.sentences,
        generated=True,
        llm_model=llm_model,
    )
