"""Console progress: tagged per-stage lines + a duration-based batch bar + one
countdown bar per in-flight file, all rendered through tqdm so scrolling log
lines never corrupt the bars.

The pipeline drives this at stage boundaries. Estimates are model-based: Whisper /
pyannote don't expose within-file progress, so a file's "time remaining" is
`audio × learned_RTF − elapsed`, refreshed once a second by a ticker thread.
Batch ETA instead uses throughput (audio processed / wall elapsed), which reflects
--jobs parallelism because it measures the whole run's wall clock.
"""
from __future__ import annotations

import sys
import threading
import time

from tqdm import tqdm


def fmt_duration(seconds: float) -> str:
    """Seconds -> 'M:SS' (under an hour) or 'H:MM:SS'."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class RateModel:
    """Learns two rates from completed files.

    - per-file RTF (wall/audio, averaged over completed files) -> per-file estimates
    - batch throughput (audio / wall-elapsed) -> batch ETA that accounts for
      parallelism, since it measures the whole run's wall clock rather than
      summing per-file times.
    """

    def __init__(self, default_rtf: float, total_audio_sec: float):
        self.default_rtf = max(0.01, default_rtf)
        self.total_audio_sec = max(0.0, total_audio_sec)
        self._sum_wall = 0.0
        self._sum_audio = 0.0
        self.completed_audio = 0.0
        self.completed_files = 0

    @property
    def per_file_rtf(self) -> float:
        if self._sum_audio > 0:
            return self._sum_wall / self._sum_audio
        return self.default_rtf

    def est_wall(self, audio_sec: float) -> float:
        return max(1.0, audio_sec * self.per_file_rtf)

    def record(self, audio_sec: float, wall_sec: float) -> None:
        if audio_sec > 0 and wall_sec > 0:
            self._sum_wall += wall_sec
            self._sum_audio += audio_sec
        self.completed_audio += max(0.0, audio_sec)
        self.completed_files += 1

    def batch_eta(self, elapsed: float) -> float | None:
        """Wall seconds left, or None until there's enough data to estimate."""
        remaining = self.total_audio_sec - self.completed_audio
        if remaining <= 0 or self.completed_audio <= 0 or elapsed <= 0:
            return None
        throughput = self.completed_audio / elapsed  # audio-sec per wall-sec
        return remaining / throughput


class SlotPool:
    """Hands out the smallest free tqdm row (>=1); row 0 is the batch bar."""

    def __init__(self):
        self._used: set[int] = set()

    def acquire(self) -> int:
        pos = 1
        while pos in self._used:
            pos += 1
        self._used.add(pos)
        return pos

    def release(self, pos: int) -> None:
        self._used.discard(pos)


class _FileState:
    __slots__ = ("idx", "total", "name", "audio_sec", "est_wall", "started", "bar", "pos")

    def __init__(self, idx, total, name, audio_sec, est_wall, started, bar, pos):
        self.idx = idx
        self.total = total
        self.name = name
        self.audio_sec = audio_sec
        self.est_wall = est_wall
        self.started = started
        self.bar = bar
        self.pos = pos


class NullReporter:
    """No-op reporter — the Pipeline default so unit tests stay silent."""

    def start_batch(self, total_audio_sec: float, file_count: int) -> None: ...
    def file_start(self, content_hash, idx, total, name, audio_sec) -> None: ...
    def stage(self, content_hash, tag, detail: str = "") -> None: ...
    def file_done(self, content_hash, wall_sec, out_path) -> None: ...
    def file_failed(self, content_hash, error) -> None: ...
    def close(self) -> None: ...


class ProgressReporter:
    """Live console reporter. When ``bars`` is False (piped output / --no-progress)
    it still prints the tagged lines but skips the live bars and the ticker thread.
    """

    def __init__(self, default_rtf: float, *, bars: bool):
        self.bars = bars
        self.default_rtf = default_rtf
        self._rate: RateModel | None = None
        self._file_count = 0
        self._slots = SlotPool()
        self._files: dict[str, _FileState] = {}
        self._batch_bar: tqdm | None = None
        self._batch_started = 0.0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None

    # --- rendering helpers ----------------------------------------------

    def _write(self, msg: str) -> None:
        tqdm.write(msg, file=sys.stderr)

    @staticmethod
    def _pct(n: float, total: float) -> int:
        return int(100 * n / total) if total > 0 else 0

    def _refresh_batch_locked(self) -> None:
        if self._batch_bar is None or self._rate is None:
            return
        total = self._rate.total_audio_sec
        done = min(self._rate.completed_audio, total)
        elapsed = time.monotonic() - self._batch_started
        eta = self._rate.batch_eta(elapsed)
        eta_str = fmt_duration(eta) if eta is not None else "…"
        self._batch_bar.n = done
        self._batch_bar.set_description_str(
            f"Audio {self._pct(done, total):3d}% {fmt_duration(done)}/{fmt_duration(total)}"
            f" · files {self._rate.completed_files}/{self._file_count}"
            f" · RTF {self._rate.per_file_rtf:.2f} · ETA {eta_str}"
        )
        self._batch_bar.refresh()

    def _refresh_file_locked(self, st: _FileState) -> None:
        elapsed = time.monotonic() - st.started
        n = min(elapsed, st.est_wall)
        if elapsed >= st.est_wall:
            tail = f"{fmt_duration(st.est_wall)}/{fmt_duration(st.est_wall)} · ~0:00 (finishing)"
        else:
            remaining = st.est_wall - elapsed
            tail = f"{fmt_duration(n)}/{fmt_duration(st.est_wall)} · ~{fmt_duration(remaining)} left"
        st.bar.n = n
        st.bar.set_description_str(
            f"[{st.idx}/{st.total}] {st.name} {self._pct(n, st.est_wall):3d}% {tail}"
        )
        st.bar.refresh()

    def _tick(self) -> None:
        while not self._stop.wait(1.0):
            with self._lock:
                for st in list(self._files.values()):
                    self._refresh_file_locked(st)
                self._refresh_batch_locked()

    # --- pipeline-facing API --------------------------------------------

    def start_batch(self, total_audio_sec: float, file_count: int) -> None:
        with self._lock:
            self._rate = RateModel(self.default_rtf, total_audio_sec)
            self._file_count = file_count
            self._batch_started = time.monotonic()
            self._write(
                f"Batch: {file_count} files · total audio {fmt_duration(total_audio_sec)} "
                f"({int(total_audio_sec)}s) · default RTF {self.default_rtf:.2f}"
            )
            if self.bars:
                self._batch_bar = tqdm(
                    total=max(1.0, total_audio_sec), position=0, leave=True,
                    bar_format="{desc} |{bar}|", file=sys.stderr,
                )
                self._refresh_batch_locked()
                self._stop.clear()
                self._ticker = threading.Thread(target=self._tick, daemon=True)
                self._ticker.start()

    def file_start(self, content_hash, idx, total, name, audio_sec) -> None:
        with self._lock:
            rate = self._rate or RateModel(self.default_rtf, 0.0)
            est = rate.est_wall(audio_sec)
            self._write(
                f"[{idx}/{total}] {name} · dur {fmt_duration(audio_sec)} · ~est {fmt_duration(est)}"
            )
            bar = None
            pos = 0
            if self.bars:
                pos = self._slots.acquire()
                bar = tqdm(
                    total=max(1.0, est), position=pos, leave=False,
                    bar_format="{desc} |{bar}|", file=sys.stderr,
                )
            self._files[content_hash] = _FileState(
                idx, total, name, audio_sec, est, time.monotonic(), bar, pos
            )

    def stage(self, content_hash, tag, detail: str = "") -> None:
        with self._lock:
            st = self._files.get(content_hash)
            name = st.name if st else content_hash
            arrow = f" → {detail}" if detail else ""
            self._write(f"[{tag:<7}] {name}{arrow}")

    def file_done(self, content_hash, wall_sec, out_path) -> None:
        with self._lock:
            st = self._files.pop(content_hash, None)
            if st is None:
                return
            if self._rate is not None:
                self._rate.record(st.audio_sec, wall_sec)
            rtf = (wall_sec / st.audio_sec) if st.audio_sec > 0 else 0.0
            self._write(
                f"[DONE {st.idx}/{st.total}] {st.name} · audio {fmt_duration(st.audio_sec)} "
                f"· took {fmt_duration(wall_sec)} · rtf {rtf:.2f} → {out_path}"
            )
            if st.bar is not None:
                st.bar.close()
                self._slots.release(st.pos)
            self._refresh_batch_locked()

    def file_failed(self, content_hash, error) -> None:
        with self._lock:
            st = self._files.pop(content_hash, None)
            if st is None:
                self._write(f"[FAIL] {content_hash}: {error}")
                return
            self._write(f"[FAIL {st.idx}/{st.total}] {st.name}: {error}")
            if st.bar is not None:
                st.bar.close()
                self._slots.release(st.pos)

    def close(self) -> None:
        with self._lock:
            self._stop.set()
        if self._ticker is not None:
            self._ticker.join(timeout=2.0)
        with self._lock:
            for st in self._files.values():
                if st.bar is not None:
                    st.bar.close()
            self._files.clear()
            if self._batch_bar is not None:
                self._batch_bar.close()
                self._batch_bar = None


def make_reporter(enabled: bool, default_rtf: float) -> ProgressReporter:
    """Build a reporter. Tagged lines always print; the live bars are on only when
    ``enabled`` (not --no-progress) AND stderr is a TTY (not piped/CI)."""
    bars = enabled and sys.stderr.isatty()
    return ProgressReporter(default_rtf, bars=bars)
