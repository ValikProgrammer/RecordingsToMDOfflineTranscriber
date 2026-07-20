"""LLM-rewritten "pretty" transcript: a non-verbatim, readable version of the talk.

Groups the verbatim transcript into ~1-minute blocks, fixes obvious mishearings and
removes filler, while keeping speakers and rough timecodes. Never invents content.
Written to out/pretty/ in addition to the normal verbatim .md (opt-in via --pretty).
"""
from __future__ import annotations

import logging

from ..config import Config
from ..models import RawDoc
from .summarize import chunk_transcript

SYSTEM_PROMPT = """You rewrite a verbatim speech-to-text transcript into a clean, readable version, in language code: {language}.
Rules:
- Group the conversation into roughly one-minute blocks. Start each block with a "[MM:SS]" timecode heading on its own line.
- Keep who said what: prefix each turn with the speaker label/name exactly as given in the input.
- Fix obvious mishearings, add punctuation, and remove filler and repetition so it is pleasant to read.
- NEVER invent facts, names, numbers, or events that are not in the transcript.
- Output GitHub-flavored Markdown only — no preamble, no commentary."""


def format_transcript_for_pretty(doc: RawDoc) -> list[str]:
    lines: list[str] = []
    for seg in doc.segments:
        m, s = divmod(int(seg.start), 60)
        speaker = f"{seg.speaker}: " if seg.speaker else ""
        lines.append(f"[{m:02d}:{s:02d}] {speaker}{seg.text}")
    return lines


def call_ollama_text(model: str, system_prompt: str, user_prompt: str, log: logging.Logger) -> str:
    import ollama

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"].strip()


def render_pretty_transcript(doc: RawDoc, cfg: Config, log: logging.Logger) -> str:
    system_prompt = SYSTEM_PROMPT.format(language=doc.language)
    lines = format_transcript_for_pretty(doc)
    chunks = chunk_transcript(lines, cfg.llm_ctx * 3)
    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            log.info(f"pretty chunk {i + 1}/{len(chunks)}")
        parts.append(call_ollama_text(cfg.llm_model, system_prompt, "\n".join(chunk), log))
    log.info(f"pretty transcript done ({cfg.llm_model})")
    return "\n\n".join(parts)
