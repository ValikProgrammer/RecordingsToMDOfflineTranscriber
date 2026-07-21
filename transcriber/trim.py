"""Silence-trim preprocessor (algorithmic, no LLM).

Two phases:
  1. detect: `python -m transcriber.trim --input-folder ./audio`
     Scans each file with ffmpeg `silencedetect`, proposes cut ranges (silence
     >= min_gap), skips files with total cut < min_total, and writes a
     trim_plan.json for the user to review/edit.
  2. apply: `python -m transcriber.trim --apply --out ./out/edited`
     Reads the (possibly edited) plan and ffmpeg-cuts each file into out/edited/.
     Originals are never modified.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from .stages.ingest import scan_audio_files

DEFAULT_NOISE_DB = -30
DEFAULT_MIN_GAP_SEC = 20.0
DEFAULT_MIN_TOTAL_SEC = 60.0
DEFAULT_PLAN = "trim_plan.json"

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


def parse_silencedetect(stderr: str) -> list[tuple[float, float]]:
    """Parse ffmpeg silencedetect stderr into (start, end) silence intervals.

    ffmpeg prints `silence_start: X` and later `silence_end: Y` lines (order
    preserved). A trailing start with no end (silence runs to EOF) is dropped
    here and handled by the caller against the known duration."""
    intervals: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in stderr.splitlines():
        m_start = _SILENCE_START_RE.search(line)
        if m_start:
            pending_start = float(m_start.group(1))
            continue
        m_end = _SILENCE_END_RE.search(line)
        if m_end and pending_start is not None:
            intervals.append((pending_start, float(m_end.group(1))))
            pending_start = None
    return intervals


def plan_file(
    silences: list[tuple[float, float]],
    duration: float,
    min_gap_sec: float,
    min_total_sec: float,
) -> dict:
    """Turn detected silences into a per-file plan: keep silences >= min_gap,
    clamp to [0, duration], and mark the file 'skip' if the total is below
    min_total. Returns {action, cuts, total_cut_sec}."""
    cuts: list[list[float]] = []
    for start, end in silences:
        start = max(0.0, start)
        end = min(duration, end) if duration > 0 else end
        if end - start >= min_gap_sec:
            cuts.append([round(start, 3), round(end, 3)])
    total = round(sum(e - s for s, e in cuts), 3)
    action = "trim" if total >= min_total_sec and cuts else "skip"
    return {"action": action, "total_cut_sec": total, "cuts": cuts if action == "trim" else []}


def keep_ranges(cuts: list[list[float]], duration: float) -> list[tuple[float, float]]:
    """Complement of the cut ranges within [0, duration] — the parts to keep."""
    ordered = sorted((max(0.0, s), min(duration, e)) for s, e in cuts if e > s)
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in ordered:
        if start > cursor:
            keep.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        keep.append((cursor, duration))
    return keep


def _aselect_expr(keeps: list[tuple[float, float]]) -> str:
    return "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in keeps)


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def detect_silences(path: Path, noise_db: int, min_gap_sec: float) -> list[tuple[float, float]]:
    # silencedetect writes to stderr; -f null discards the decoded audio
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_gap_sec}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return parse_silencedetect(proc.stderr)


def build_plan(folder: Path, noise_db: int, min_gap_sec: float, min_total_sec: float, log=print) -> dict:
    files = scan_audio_files(folder)
    entries = []
    for path in files:
        duration = probe_duration(path)
        silences = detect_silences(path, noise_db, min_gap_sec)
        fileplan = plan_file(silences, duration, min_gap_sec, min_total_sec)
        entries.append({"file": path.name, "duration_sec": round(duration, 3), **fileplan})
        log(f"{path.name}: {fileplan['action']} (cut {fileplan['total_cut_sec']:.0f}s of {duration:.0f}s)")
    return {
        "params": {"noise_db": noise_db, "min_gap_sec": min_gap_sec, "min_total_sec": min_total_sec},
        "files": entries,
    }


def apply_cut(src: Path, cuts: list[list[float]], duration: float, dst: Path) -> None:
    keeps = keep_ranges(cuts, duration)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", "-i", str(src),
         "-af", f"aselect='{_aselect_expr(keeps)}',asetpts=N/SR/TB", str(dst)],
        check=True, capture_output=True, text=True,
    )


def apply_plan(plan: dict, input_folder: Path, out_dir: Path, log=print) -> int:
    applied = 0
    for entry in plan.get("files", []):
        if entry.get("action") != "trim" or not entry.get("cuts"):
            continue
        src = input_folder / entry["file"]
        if not src.exists():
            log(f"skip {entry['file']}: not found in {input_folder}")
            continue
        dst = out_dir / entry["file"]
        apply_cut(src, entry["cuts"], entry["duration_sec"], dst)
        log(f"trimmed {entry['file']} -> {dst}")
        applied += 1
    return applied


def fmt_elapsed(seconds: float) -> str:
    """Human-readable elapsed time, e.g. '42s' or '3m 42s'."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transcriber.trim", description="Detect and cut silence (algorithmic)")
    parser.add_argument("--input-folder", "--input", dest="input_folder", default="./audio")
    parser.add_argument("--plan", dest="plan", default=DEFAULT_PLAN)
    parser.add_argument("--apply", dest="apply", action="store_true")
    parser.add_argument("--out", dest="out", default="./out/edited")
    parser.add_argument("--noise-db", dest="noise_db", type=int, default=DEFAULT_NOISE_DB)
    parser.add_argument("--min-gap", dest="min_gap", type=float, default=DEFAULT_MIN_GAP_SEC)
    parser.add_argument("--min-total", dest="min_total", type=float, default=DEFAULT_MIN_TOTAL_SEC)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    plan_path = Path(args.plan)
    if args.apply:
        if not plan_path.exists():
            print(f"Plan not found: {plan_path}. Run detect first.")
            return 1
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        applied = apply_plan(plan, Path(args.input_folder), Path(args.out))
        print(f"Applied {applied} trim(s) -> {args.out}")
        print(f"Done in {fmt_elapsed(time.perf_counter() - start)}")
        return 0

    plan = build_plan(Path(args.input_folder), args.noise_db, args.min_gap, args.min_total)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    to_trim = sum(1 for f in plan["files"] if f["action"] == "trim")
    print(f"Wrote {plan_path} — {to_trim}/{len(plan['files'])} file(s) proposed for trimming.")
    print("Review/edit it (drop pairs, adjust timecodes, set action=skip), then run --apply.")
    print(f"Done in {fmt_elapsed(time.perf_counter() - start)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
