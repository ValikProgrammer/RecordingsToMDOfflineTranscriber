"""Auto language detection from sampled audio windows (issue #25).

Whisper's built-in detection only looks at the FIRST 30s of a file — often
intro, silence, or noise, so it guesses wrong. Instead we sample up to 3 windows
(start / middle / end), detect the language of each with a confidence
probability, drop low-confidence (noisy) windows, and decide:
  - all confident windows agree  -> force that language
  - they disagree                -> bilingual/uncertain, return None (don't force;
                                    let the backend switch language per segment)
  - none confident               -> return None (fall back to the backend default)

Detection uses faster-whisper's detect_language (fast, returns a probability),
independent of the transcription backend. If faster-whisper isn't installed, we
log and return None so behaviour degrades to the backend's built-in detection.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
from pathlib import Path

WINDOW_SEC = 30
DETECT_MODEL = "large-v3"

# The detector runs on CPU while mlx transcribes a different file on the GPU
# (detection lives in pipeline stage A, transcription in stage B — they overlap).
# Cache one model for the whole batch and serialize inference: loading per file
# would waste ~1.5GB each, and CTranslate2 calls are serialized to be safe.
_detector = None
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


def _get_detector(log: logging.Logger):
    global _detector
    with _load_lock:
        if _detector is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                log.info("lang-detect: faster-whisper not installed; skipping (backend will auto-detect)")
                return None
            _detector = WhisperModel(DETECT_MODEL, device="cpu", compute_type="int8")
    return _detector


def window_starts(duration: float, num_windows: int, window_sec: int = WINDOW_SEC) -> list[float]:
    """Start offsets for up to `num_windows` non-overlapping windows across the file.

    Short files collapse to fewer windows (deduped); a file shorter than one
    window yields a single window at 0.
    """
    if duration <= window_sec or num_windows <= 1:
        return [0.0]
    # Space windows so each is centred in its share of the timeline, clamped in-range.
    last_start = max(0.0, duration - window_sec)
    starts = [min(last_start, (i + 0.5) * duration / num_windows - window_sec / 2) for i in range(num_windows)]
    starts = [max(0.0, s) for s in starts]
    deduped: list[float] = []
    for s in starts:
        if not deduped or abs(s - deduped[-1]) >= 1.0:
            deduped.append(s)
    return deduped


def decide_language(detections: list[tuple[str, float] | None], min_prob: float) -> str | None:
    """Pure vote: force a language only if every confident window agrees on it."""
    confident = [d for d in detections if d is not None and d[1] >= min_prob]
    langs = {lang for lang, _ in confident}
    if len(langs) == 1:
        return confident[0][0]
    return None  # 0 confident (noise) or >1 (bilingual/uncertain) -> don't force


def _extract_window(wav: Path, start: float, out: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", str(WINDOW_SEC), "-i", str(wav),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)],
        check=True, capture_output=True,
    )
    return out


def _probe_duration(wav: Path) -> float:
    from .audio import probe_duration

    return probe_duration(wav)


def _detect_windows(windows: list[Path], log: logging.Logger) -> list[tuple[str, float] | None]:
    model = _get_detector(log)
    if model is None:
        return [None] * len(windows)
    from faster_whisper import decode_audio

    results: list[tuple[str, float] | None] = []
    for w in windows:
        try:
            with _infer_lock:  # CTranslate2 model is shared across stage-A threads
                lang, prob, _ = model.detect_language(decode_audio(str(w)))
            log.info(f"lang-detect: window -> {lang} (p={prob:.2f})")
            results.append((lang, prob))
        except Exception as exc:  # noqa: BLE001 - one bad window shouldn't kill detection
            log.info(f"lang-detect: window failed ({exc})")
            results.append(None)
    return results


def detect_language(
    wav: Path, log: logging.Logger, *, min_prob: float = 0.6, num_windows: int = 3
) -> str | None:
    """Return a language code to force, or None to leave the backend to auto-detect."""
    duration = _probe_duration(wav)
    starts = window_starts(duration, num_windows)
    with tempfile.TemporaryDirectory(prefix="langdetect-") as td:
        windows = [_extract_window(wav, s, Path(td) / f"win{i}.wav") for i, s in enumerate(starts)]
        detections = _detect_windows(windows, log)
    decision = decide_language(detections, min_prob)
    log.info(f"lang-detect: decision={decision or 'auto (no force)'}")
    return decision
