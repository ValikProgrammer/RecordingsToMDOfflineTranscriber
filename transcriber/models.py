"""Shared dataclasses for the transcriber pipeline (see CREATE_SYSTEM.md §5)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Word:
    w: str
    start: float
    end: float
    speaker: Optional[str] = None


@dataclass
class Segment:
    start: float
    end: float
    speaker: Optional[str]
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class SpeakerMeta:
    label: str
    name: Optional[str] = None
    embedding: Optional[list[float]] = None
    total_speech_sec: float = 0.0


@dataclass
class TopicRef:
    term: str
    ts: float


@dataclass
class KeyTopic:
    topic: str
    ts: float
    note: str = ""


@dataclass
class Decision:
    text: str
    ts: float


@dataclass
class Summary:
    title: str
    text: str
    topics: list[TopicRef] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    key_topics: list[KeyTopic] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    length_tier: str = ""
    generated: bool = False
    llm_model: str = ""


@dataclass
class AsrInfo:
    backend: str
    model: str
    turbo: bool


@dataclass
class RawDoc:
    schema: int
    content_hash: str
    source_name: str
    source_path: str
    language: str
    duration_sec: float
    num_speakers: int
    is_monologue: bool
    asr: AsrInfo
    created_at: str
    segments: list[Segment] = field(default_factory=list)
    speakers_meta: list[SpeakerMeta] = field(default_factory=list)
    summary: Optional[Summary] = None

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "content_hash": self.content_hash,
            "source_name": self.source_name,
            "source_path": self.source_path,
            "language": self.language,
            "duration_sec": self.duration_sec,
            "num_speakers": self.num_speakers,
            "is_monologue": self.is_monologue,
            "asr": {"backend": self.asr.backend, "model": self.asr.model, "turbo": self.asr.turbo},
            "created_at": self.created_at,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "speaker": s.speaker,
                    "text": s.text,
                    "words": [
                        {"w": w.w, "start": w.start, "end": w.end, "speaker": w.speaker}
                        for w in s.words
                    ],
                }
                for s in self.segments
            ],
            "speakers_meta": [
                {
                    "label": sm.label,
                    "name": sm.name,
                    "embedding": sm.embedding,
                    "total_speech_sec": sm.total_speech_sec,
                }
                for sm in self.speakers_meta
            ],
            "summary": _summary_to_dict(self.summary) if self.summary else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RawDoc":
        return cls(
            schema=data["schema"],
            content_hash=data["content_hash"],
            source_name=data["source_name"],
            source_path=data["source_path"],
            language=data["language"],
            duration_sec=data["duration_sec"],
            num_speakers=data["num_speakers"],
            is_monologue=data["is_monologue"],
            asr=AsrInfo(**data["asr"]),
            created_at=data["created_at"],
            segments=[
                Segment(
                    start=s["start"],
                    end=s["end"],
                    speaker=s["speaker"],
                    text=s["text"],
                    words=[Word(**w) for w in s.get("words", [])],
                )
                for s in data.get("segments", [])
            ],
            speakers_meta=[SpeakerMeta(**sm) for sm in data.get("speakers_meta", [])],
            summary=_summary_from_dict(data["summary"]) if data.get("summary") else None,
        )


def _summary_to_dict(summary: Summary) -> dict:
    return {
        "title": summary.title,
        "text": summary.text,
        "topics": [{"term": t.term, "ts": t.ts} for t in summary.topics],
        "hashtags": list(summary.hashtags),
        "key_topics": [
            {"topic": kt.topic, "ts": kt.ts, "note": kt.note} for kt in summary.key_topics
        ],
        "decisions": [{"text": d.text, "ts": d.ts} for d in summary.decisions],
        "length_tier": summary.length_tier,
        "generated": summary.generated,
        "llm_model": summary.llm_model,
    }


def _summary_from_dict(data: dict) -> Summary:
    return Summary(
        title=data["title"],
        text=data["text"],
        topics=[TopicRef(**t) for t in data.get("topics", [])],
        hashtags=list(data.get("hashtags", [])),
        key_topics=[KeyTopic(**kt) for kt in data.get("key_topics", [])],
        decisions=[Decision(**d) for d in data.get("decisions", [])],
        length_tier=data.get("length_tier", ""),
        generated=data.get("generated", False),
        llm_model=data.get("llm_model", ""),
    )


@dataclass
class AsrWord:
    w: str
    start: float
    end: float


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    words: list[AsrWord] = field(default_factory=list)


@dataclass
class AsrResult:
    language: str
    segments: list[AsrSegment]
    backend: str
    model: str
    turbo: bool


@dataclass
class DiarSegment:
    start: float
    end: float
    speaker: str


@dataclass
class DiarResult:
    segments: list[DiarSegment] = field(default_factory=list)
    embeddings: dict[str, list[float]] = field(default_factory=dict)
    total_speech_sec: dict[str, float] = field(default_factory=dict)


STAGE_NAMES = ("text", "diarize", "summary", "pretty")


@dataclass
class StageState:
    status: str  # "pending" | "in_progress" | "done" | "skipped" | "failed"
    updated_at: str = ""
    reason: Optional[str] = None


def default_stages() -> dict[str, StageState]:
    return {name: StageState(status="pending") for name in STAGE_NAMES}


@dataclass
class FileTask:
    path: object  # pathlib.Path; kept loosely typed to avoid an import cycle
    content_hash: str
    source_name: str
    status: str  # "to_do" | "skip" | "redo"
    reason: str
    audio_sec: float = 0.0  # ffprobe'd up front to size the progress bar (0 = unknown)


@dataclass
class ManifestEntry:
    content_hash: str
    source_name: str
    status: str  # "done" | "failed" | "in_progress"
    language: Optional[str] = None
    num_speakers: Optional[int] = None
    duration_sec: Optional[float] = None
    out_path: Optional[str] = None
    raw_path: Optional[str] = None
    log_path: Optional[str] = None
    elapsed_sec: Optional[float] = None
    error: Optional[str] = None
    updated_at: str = ""
    stages: dict[str, StageState] = field(default_factory=default_stages)


def stage_status(entry: ManifestEntry, name: str) -> str:
    return entry.stages[name].status


def set_stage(entry: ManifestEntry, name: str, status: str, reason: Optional[str] = None) -> ManifestEntry:
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry.stages[name] = StageState(status=status, updated_at=updated_at, reason=reason)
    return entry
