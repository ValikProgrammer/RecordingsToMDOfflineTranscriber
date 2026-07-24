"""Filename logic: technical vs meaningful names, dates, sanitization,
collisions, transliteration for log slugs, tag slugification (CREATE_SYSTEM.md §8)."""
from __future__ import annotations

import re
import subprocess
from datetime import date, datetime
from pathlib import Path

# --- §8.1 Technical vs meaningful name detection -----------------------------

_TECHNICAL_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),  # 2026-07-12
    re.compile(r"^\d{8}$"),  # 20260712
    re.compile(r"^\d{4}\.\d{2}\.\d{2}$"),  # 2026.07.12
    re.compile(r"^\d{2}-\d{2}-\d{4}$"),  # 12-07-2026
    re.compile(r"^(rec|aud)[_\-]?\d{6,8}$", re.IGNORECASE),  # REC_20260712, AUD-20260712
    re.compile(r"^(voice|recording)[_\s]*\d*$", re.IGNORECASE),  # voice 001, recording_003
    re.compile(r"^(new recording|новая запись|recording|запись)(\s+\d+)?$", re.IGNORECASE),
    re.compile(r"^(audio|voice memo)$", re.IGNORECASE),  # junk
    re.compile(r"^[\d_\-\s]+$"),  # only digits/underscores
]


def is_technical_name(stem: str) -> bool:
    """True if `stem` (filename without extension) looks machine-generated.

    Extensible list of regexes — the simplest approach per §8.1.
    """
    normalized = stem.strip()
    return any(p.match(normalized) for p in _TECHNICAL_PATTERNS)


# --- §8.2 Date extraction -----------------------------------------------------

# Year to assume for month-name device auto-names that carry no year
# (iOS voice-memo style "Dec 6, 23.57" — day/month only). Recordings are from
# the recent past, so a fixed default is used and the true year, when it matters,
# comes from creation_time/filesystem via the oldest-date rule in resolve_date.
DEFAULT_YEAR = 2025

_DATE_PATTERNS = [
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})"), lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"(\d{4})(\d{2})(\d{2})"), lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"(\d{2})-(\d{2})-(\d{4})"), lambda m: date(int(m[3]), int(m[2]), int(m[1]))),
]

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
# "Dec 6", "December 6", "Sep 2, 16.23", "Dec 6, 2024" — day is required right
# after the month; an optional 4-digit year is honored, otherwise DEFAULT_YEAR.
_MONTH_NAME_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:,?\s*(\d{4}))?",
    re.IGNORECASE,
)


def extract_date_from_name(stem: str) -> date | None:
    for pattern, build in _DATE_PATTERNS:
        m = pattern.search(stem)
        if m:
            try:
                return build(m)
            except ValueError:
                continue
    m = _MONTH_NAME_RE.search(stem)
    if m:
        month = _MONTHS[m.group(1).lower()]
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else DEFAULT_YEAR
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


def extract_creation_time(path: Path) -> date | None:
    """The recording's true date from the audio container's `creation_time` tag
    (via ffprobe). Returns None if ffprobe/tag is missing — never raises."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", out)
    if not m:
        return None
    try:
        return date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return None


def _filesystem_dates(path: Path) -> list[date]:
    stat = path.stat()
    dates: list[date] = []
    birthtime = getattr(stat, "st_birthtime", None)
    if birthtime:
        dates.append(datetime.fromtimestamp(birthtime).date())
    dates.append(datetime.fromtimestamp(stat.st_mtime).date())
    return dates


def extract_date_from_file(path: Path) -> date:
    stat = path.stat()
    ts = getattr(stat, "st_birthtime", None) or stat.st_mtime
    return datetime.fromtimestamp(ts).date()


_UNSET = object()


def resolve_date(stem: str, path: Path, creation_time=_UNSET) -> date:
    """The recording date, taken as the OLDEST of every available signal — the
    filename date, the container `creation_time`, and the filesystem birth/mtime.

    Copying a recording bumps its filesystem timestamps to the copy date, so the
    oldest candidate is the one closest to when it was actually recorded.
    `creation_time` is probed via ffprobe unless passed in (for tests)."""
    candidates: list[date] = []
    name_date = extract_date_from_name(stem)
    if name_date is not None:
        candidates.append(name_date)
    ct = extract_creation_time(path) if creation_time is _UNSET else creation_time
    if ct is not None:
        candidates.append(ct)
    candidates.extend(_filesystem_dates(path))
    return min(candidates)


# --- §8.3 Title -----------------------------------------------------------

_SEPARATORS = re.compile(r"[_\-.]+")


def normalize_title_from_name(stem: str, max_words: int = 4) -> str:
    spaced = _SEPARATORS.sub(" ", stem)
    words = [w for w in spaced.split() if w][:max_words]
    if not words:
        return "Recording"
    words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words)


def default_title_for_date(day: date) -> str:
    return f"Recording {day.isoformat()}"


# --- §8.4 Filename sanitization / collisions ---------------------------------

# Filesystem-forbidden + Obsidian-forbidden (# ^ [ ]) + control chars.
_FORBIDDEN_CHARS = re.compile(r'[\\/:*?"<>|#^\[\]\x00-\x1f]')
_MULTI_SPACE = re.compile(r"\s+")


def sanitize_filename_component(text: str, max_len: int = 60) -> str:
    cleaned = _FORBIDDEN_CHARS.sub("", text)
    cleaned = _MULTI_SPACE.sub(" ", cleaned).strip()
    return cleaned[:max_len].rstrip()


def build_output_filename(day: date, title: str) -> str:
    safe_title = sanitize_filename_component(title)
    return f"{day.isoformat()} — {safe_title}.md"


def resolve_collision(out_folder: Path, filename: str) -> Path:
    """Atomically reserve the first free path for `filename`, appending ` (N)`.

    Creates the file as an empty placeholder (O_CREAT|O_EXCL) before returning,
    so two concurrent callers can never be handed the same path — a plain
    exists()-then-use check races and lets two recordings collide onto one file
    (issue #17). The caller overwrites the placeholder via os.replace."""
    out_folder.mkdir(parents=True, exist_ok=True)
    candidate = out_folder / filename
    stem, suffix = candidate.stem, candidate.suffix
    n = 1
    while True:
        try:
            candidate.touch(exist_ok=False)  # O_CREAT|O_EXCL: reserve atomically
            return candidate
        except FileExistsError:
            n += 1
            candidate = out_folder / f"{stem} ({n}){suffix}"


# --- Transliteration for per-file log slugs (§5.3 example: rabochaya-vstrecha) ---

_RU_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def transliterate_ru(text: str) -> str:
    return "".join(_RU_TO_LATIN.get(ch, ch) for ch in text.lower())


_LOG_SAFE = re.compile(r"[^a-z0-9]+")


def slugify_for_log(name: str) -> str:
    translit = transliterate_ru(name)
    slug = _LOG_SAFE.sub("-", translit).strip("-")
    return slug or "file"


# --- Obsidian tag slugification (keeps Cyrillic, per §11) --------------------

_TAG_UNSAFE = re.compile(r"[^\w\-]+", re.UNICODE)


def slugify_tag(text: str) -> str | None:
    lowered = re.sub(r"\s+", "-", text.strip().lower().lstrip("#"))
    lowered = _TAG_UNSAFE.sub("", lowered)
    if not lowered or lowered.replace("-", "").isdigit():
        return None
    return lowered
