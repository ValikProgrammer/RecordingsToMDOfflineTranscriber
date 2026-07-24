"""LLM-rewritten "pretty" transcript: a non-verbatim, readable version of the talk.

Groups the verbatim transcript into topic-based blocks (each headed by the earliest
timecode of that topic), fixes obvious mishearings and strips filler, while preserving
~80% of the actual content — every distinct subject, fact and number is kept. Never
invents content. Written as a full document (frontmatter + summary + the rewritten
transcript) to out/pretty/ in addition to the normal verbatim .md (opt-in via --pretty).
"""
from __future__ import annotations

import logging

from ..config import Config
from ..models import RawDoc
from .summarize import chunk_transcript

SYSTEM_PROMPT = """You rewrite a verbatim speech-to-text transcript into a clean, readable version, in language code: {language}.

This is NOT a summary. Your job is to make the SAME talk pleasant to read, not to shorten it. Keep roughly 80% of the actual content. Preserve every distinct subject, fact, name, number, reason, and nuance that was actually said — never collapse several subjects into one clause (e.g. do not turn a real discussion of healthcare, crypto, and culture into "discussed crypto and culture" — each subject stays, with its details).

Rules:
- Structure the text as blocks, one per distinct topic or episode of the conversation (not fixed time windows).
- Start each block with a "[MM:SS]" timecode heading on its own line, using the EARLIEST timecode of that topic taken from the input. Timecodes are mandatory — every block must have one. When you merge adjacent sentences that belong to the same topic, keep the earliest of their timecodes.
- Prefix a turn with the speaker label/name (exactly as given in the input) only when several speakers appear in a block or it reads as a dialogue; for a single-speaker stretch, no prefix is needed.
- Fix obvious mishearings, add punctuation, and remove only filler words and pure repetition ("ну", "типа", "вот", "короче", restarts) — do NOT remove content.
- This is an automatic transcription: nonsense, or the same word/phrase repeated many times in one place, is usually recognition noise rather than real speech — drop it. Reason logically about the context: if a common word does not fit, it was probably misheard, so replace it with what was most likely actually said. Apply this ONLY to ordinary, everyday words. Leave domain-specific or unusual terms as-is even if they look wrong — you cannot reliably guess them, so keep them for the user to fix.
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
