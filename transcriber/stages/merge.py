"""Word<->speaker assignment, reply grouping, monologue collapse, --names (§9)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..models import (
    AsrInfo,
    AsrResult,
    DiarResult,
    RawDoc,
    Segment,
    SpeakerMeta,
    Word,
)

PAUSE_THRESHOLD_SEC = 1.5


def assign_speaker(word_start: float, word_end: float, diar: DiarResult) -> str | None:
    if not diar.segments:
        return None
    best_speaker = None
    best_overlap = 0.0
    for seg in diar.segments:
        overlap = min(word_end, seg.end) - max(word_start, seg.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg.speaker
    if best_speaker is not None:
        return best_speaker
    nearest = min(
        diar.segments,
        key=lambda s: min(abs(s.start - word_end), abs(s.end - word_start)),
    )
    return nearest.speaker


def build_segments(asr: AsrResult, diar: DiarResult) -> list[Segment]:
    flat_words: list[Word] = []
    for asr_seg in asr.segments:
        for w in asr_seg.words:
            speaker = assign_speaker(w.start, w.end, diar)
            flat_words.append(Word(w=w.w, start=w.start, end=w.end, speaker=speaker))

    segments: list[Segment] = []
    current: list[Word] = []
    for word in flat_words:
        if current and (
            word.speaker != current[-1].speaker
            or word.start - current[-1].end > PAUSE_THRESHOLD_SEC
        ):
            segments.append(_flush(current))
            current = []
        current.append(word)
    if current:
        segments.append(_flush(current))
    return segments


def _flush(words: list[Word]) -> Segment:
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        speaker=words[0].speaker,
        text=" ".join(w.w for w in words),
        words=words,
    )


def compute_total_speech(segments: list[Segment]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for seg in segments:
        if seg.speaker is None:
            continue
        totals[seg.speaker] = totals.get(seg.speaker, 0.0) + (seg.end - seg.start)
    return totals


def collapse_monologue(segments: list[Segment], mono_threshold: float) -> bool:
    """True if the whole recording should be treated as a single-speaker monologue:
    either exactly one raw speaker, or one speaker dominates >= mono_threshold of
    total speech (protects against a false second speaker from noise/echo)."""
    totals = compute_total_speech(segments)
    if not totals:
        return False
    if len(totals) == 1:
        return True
    dominant_share = max(totals.values()) / sum(totals.values())
    return dominant_share >= mono_threshold


def apply_names(segments: list[Segment], names: list[str] | None) -> dict[str, str]:
    """Map raw SPEAKER_XX labels to --names entries by order of first appearance."""
    if not names:
        return {}
    first_seen: dict[str, float] = {}
    for seg in segments:
        if seg.speaker is not None and seg.speaker not in first_seen:
            first_seen[seg.speaker] = seg.start
    ordered_speakers = sorted(first_seen, key=lambda s: first_seen[s])
    return {speaker: names[i] for i, speaker in enumerate(ordered_speakers) if i < len(names)}


def _apply_name_map(segments: list[Segment], name_map: dict[str, str]) -> list[Segment]:
    result = []
    for seg in segments:
        new_speaker = name_map.get(seg.speaker, seg.speaker) if seg.speaker else seg.speaker
        words = [
            Word(
                w=w.w, start=w.start, end=w.end,
                speaker=name_map.get(w.speaker, w.speaker) if w.speaker else w.speaker,
            )
            for w in seg.words
        ]
        result.append(Segment(start=seg.start, end=seg.end, speaker=new_speaker, text=seg.text, words=words))
    return result


def _strip_unnamed_raw_labels(segments: list[Segment]) -> list[Segment]:
    """For monologue docs: drop speaker labels that are still raw SPEAKER_XX
    (i.e. weren't given a name via --names); named speakers keep their name."""
    result = []
    for seg in segments:
        if seg.speaker is not None and seg.speaker.startswith("SPEAKER_"):
            words = [Word(w=w.w, start=w.start, end=w.end, speaker=None) for w in seg.words]
            result.append(Segment(start=seg.start, end=seg.end, speaker=None, text=seg.text, words=words))
        else:
            result.append(seg)
    return result


def fold_phantom_speakers(segments: list[Segment], min_share: float) -> list[Segment]:
    """Reassign segments of tiny "phantom" speakers to the nearest real speaker.

    Diarization sometimes invents an extra speaker from noise/short bursts. Any
    speaker whose share of total speech is below `min_share` is folded into the
    nearest real speaker by time. `min_share <= 0` disables this (no-op)."""
    if min_share <= 0:
        return segments
    totals = compute_total_speech(segments)
    if len(totals) <= 1:
        return segments
    total = sum(totals.values())
    if total <= 0:
        return segments
    real = {sp for sp, sec in totals.items() if sec / total >= min_share}
    if not real or len(real) == len(totals):
        return segments  # nothing to fold, or folding would remove every speaker
    real_segments = [s for s in segments if s.speaker in real]

    def nearest_real_speaker(seg: Segment) -> str:
        best = min(real_segments, key=lambda r: min(abs(r.start - seg.start), abs(r.end - seg.end)))
        return best.speaker

    result: list[Segment] = []
    for s in segments:
        if s.speaker is not None and s.speaker not in real:
            new_speaker = nearest_real_speaker(s)
            words = [Word(w=w.w, start=w.start, end=w.end, speaker=new_speaker if w.speaker else w.speaker) for w in s.words]
            result.append(Segment(start=s.start, end=s.end, speaker=new_speaker, text=s.text, words=words))
        else:
            result.append(s)
    return result


def merge(
    asr: AsrResult,
    diar: DiarResult,
    mono_threshold: float,
    names: list[str] | None,
    log: logging.Logger,
    *,
    content_hash: str,
    source_name: str,
    source_path: str,
    duration_sec: float,
    min_speaker_share: float = 0.0,
) -> RawDoc:
    segments = build_segments(asr, diar)
    segments = fold_phantom_speakers(segments, min_speaker_share)
    is_monologue = collapse_monologue(segments, mono_threshold)
    totals = compute_total_speech(segments)

    name_map = apply_names(segments, names)
    if name_map:
        segments = _apply_name_map(segments, name_map)
    if is_monologue:
        segments = _strip_unnamed_raw_labels(segments)

    num_speakers = 1 if is_monologue else len(totals)
    speakers_meta = [
        SpeakerMeta(
            label=label,
            name=name_map.get(label),
            embedding=diar.embeddings.get(label),
            total_speech_sec=totals.get(label, 0.0),
        )
        for label in totals
    ]

    log.info(f"merge done: num_speakers={num_speakers}, monologue={is_monologue}")

    return RawDoc(
        schema=1,
        content_hash=content_hash,
        source_name=source_name,
        source_path=source_path,
        language=asr.language,
        duration_sec=duration_sec,
        num_speakers=num_speakers,
        is_monologue=is_monologue,
        asr=AsrInfo(backend=asr.backend, model=asr.model, turbo=asr.turbo),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        segments=segments,
        speakers_meta=speakers_meta,
        summary=None,
    )


def build_text_doc(
    asr: AsrResult,
    *,
    content_hash: str,
    source_name: str,
    source_path: str,
    duration_sec: float,
) -> RawDoc:
    """Build a RawDoc for --text mode: no diarization, no roles at all (§16 milestone 4)."""
    segments = [
        Segment(
            start=s.start,
            end=s.end,
            speaker=None,
            text=s.text,
            words=[Word(w=w.w, start=w.start, end=w.end, speaker=None) for w in s.words],
        )
        for s in asr.segments
    ]
    return RawDoc(
        schema=1,
        content_hash=content_hash,
        source_name=source_name,
        source_path=source_path,
        language=asr.language,
        duration_sec=duration_sec,
        num_speakers=0,
        is_monologue=True,
        asr=AsrInfo(backend=asr.backend, model=asr.model, turbo=asr.turbo),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        segments=segments,
        speakers_meta=[],
        summary=None,
    )
