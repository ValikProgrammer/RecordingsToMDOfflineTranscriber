"""Rename post-pass: LLM decides which generated .md docs have poor/auto-generated
titles and proposes better ones. No templates — the model makes every call.

Three stages, one evolving rename_plan.json (hand-editable between stages):
  1. classify: `python -m transcriber.rename --classify --folder ./out`
     LLM sees ONLY filenames (cheap) and flags which need renaming.
  2. propose:  `python -m transcriber.rename --propose`
     For the flagged subset only, LLM sees name + summary + topics and proposes
     a new title; a new filename is built from it (date kept from the old name).
  3. apply:    `python -m transcriber.rename --apply`
     Renames the .md (collision-safe), rewrites the in-doc Title/heading, and
     renames + retitles the out/pretty/ twin. Source audio is never touched.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path

from . import naming
from .config import load_config
from .stages.render import yaml_escape
from .stages.summarize import call_ollama_json

DEFAULT_FOLDER = "./out"
DEFAULT_PLAN = "rename_plan.json"
DEFAULT_PRETTY_SUBDIR = "pretty"
DEFAULT_BATCH_SIZE = 576

CLASSIFY_SYSTEM = (
    "You are given a numbered list of transcript filenames (one per Excel-style "
    "column label A, B, C, ... AA, AB, ...). Some are meaningful human titles; "
    "others are auto-generated device names (e.g. 'Dec 6, 23 57', 'New Recording 5', "
    "'2026-07-20', 'voice 001') or otherwise poor and worth rewriting. "
    "Respond ONLY with a single JSON object: {\"rename\": [labels]} listing the "
    "column labels whose filename should be rewritten. Meaningful names are omitted."
)

PROPOSE_SYSTEM = (
    "For each item (Excel-style column label A, B, C, ...) you get the current "
    "filename, a summary, and topics of a transcript. Propose a concise, meaningful "
    "title (3-8 words, no date, no file extension) in the same language as the "
    "content. Respond ONLY with a single JSON object mapping each label to its new "
    "title, e.g. {\"A\": \"Discussion of apostille paperwork\"}."
)


# --- Excel-style column labels ------------------------------------------------

def excel_label(i: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, 27 -> AB (spreadsheet column naming).

    Numeric indices confuse the model in practice; letter labels are more reliable.
    """
    label = ""
    n = i + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


def _batched(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# --- .md scanning / parsing ---------------------------------------------------

def scan_md_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.md") if p.is_file())


_TOPIC_RE = re.compile(r"^- \[[^\]]*\]\s*(.+)$")


def parse_summary_and_topics(text: str) -> tuple[str, list[str]]:
    """Extract the '### Summary' paragraph and the '**Topics:**' terms (§ render)."""
    lines = text.splitlines()
    summary = ""
    for i, line in enumerate(lines):
        if line.strip() == "### Summary":
            buf: list[str] = []
            for l in lines[i + 1 :]:
                s = l.strip()
                if s == "":
                    if buf:
                        break
                    continue
                if s.startswith("#") or s.startswith("**"):
                    break
                buf.append(s)
            summary = " ".join(buf)
            break

    topics: list[str] = []
    in_topics = False
    for line in lines:
        s = line.strip()
        if s.startswith("**Topics:**"):
            in_topics = True
            continue
        if in_topics:
            m = _TOPIC_RE.match(s)
            if m:
                topics.append(m.group(1).strip())
            elif s == "":
                continue
            else:
                break
    return summary, topics


# --- stage 1: classify --------------------------------------------------------

def classify_files(
    names: list[str], model: str, log: logging.Logger, batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[str, bool]:
    """Map each filename -> should_rename, via cheap filename-only LLM batches."""
    decisions: dict[str, bool] = {}
    for batch in _batched(names, batch_size):
        labels = {excel_label(i): name for i, name in enumerate(batch)}
        user = "\n".join(f"{lab}. {name}" for lab, name in labels.items())
        raw = call_ollama_json(model, CLASSIFY_SYSTEM, user, log)
        flagged = set(raw.get("rename", []))
        for lab, name in labels.items():
            decisions[name] = lab in flagged
    return decisions


def build_classify_plan(
    folder: Path, model: str, log: logging.Logger, batch_size: int
) -> dict:
    names = [p.name for p in scan_md_files(folder)]
    decisions = classify_files(names, model, log, batch_size)
    files = [
        {
            "file": name,
            "action": "rename" if decisions.get(name) else "keep",
            "reason": "auto-generated / poor name" if decisions.get(name) else "meaningful",
        }
        for name in names
    ]
    return {"folder": str(folder), "files": files}


# --- stage 2: propose ---------------------------------------------------------

def propose_titles(
    entries: list[dict], model: str, log: logging.Logger, batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[str, str]:
    """entries: [{name, summary, topics}] -> {name: new_title}."""
    titles: dict[str, str] = {}
    for batch in _batched(entries, batch_size):
        labels = {excel_label(i): e for i, e in enumerate(batch)}
        lines = [
            f"{lab}. name={e['name']} | summary={e['summary']} | topics={'; '.join(e['topics'])}"
            for lab, e in labels.items()
        ]
        raw = call_ollama_json(model, PROPOSE_SYSTEM, "\n".join(lines), log)
        for lab, e in labels.items():
            proposed = raw.get(lab)
            if proposed:
                titles[e["name"]] = str(proposed).strip()
    return titles


_FRONTMATTER_DATE_RE = re.compile(r"^Date:\s*(\d{4})-(\d{2})-(\d{2})\s*$", re.M)


def parse_frontmatter_date(text: str) -> date | None:
    """The canonical `Date:` from the doc's Obsidian frontmatter, if present."""
    m = _FRONTMATTER_DATE_RE.search(text)
    if not m:
        return None
    try:
        return date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return None


def resolve_date(name: str, folder: Path, text: str | None) -> date | None:
    """Date for the renamed file, algorithmically (never from the LLM):
    frontmatter `Date:` (from Obsidian) -> date in the current filename -> file mtime.
    """
    if text:
        day = parse_frontmatter_date(text)
        if day is not None:
            return day
    day = naming.extract_date_from_name(Path(name).stem)
    if day is not None:
        return day
    path = folder / name
    return naming.extract_date_from_file(path) if path.exists() else None


def fill_proposals(
    plan: dict, folder: Path, model: str, log: logging.Logger, batch_size: int
) -> dict:
    to_rename = [e for e in plan["files"] if e.get("action") == "rename"]
    entries = []
    for e in to_rename:
        path = folder / e["file"]
        summary, topics, text = "", [], None
        if path.exists():
            text = path.read_text(encoding="utf-8")
            summary, topics = parse_summary_and_topics(text)
        entries.append(
            {"name": e["file"], "summary": summary, "topics": topics,
             "day": resolve_date(e["file"], folder, text)}
        )

    titles = propose_titles(entries, model, log, batch_size)
    by_name = {en["name"]: en for en in entries}
    for e in to_rename:
        title = titles.get(e["file"])
        if not title:
            continue
        day = by_name[e["file"]]["day"]
        if day is None:
            log.info(f"skip {e['file']}: no date resolvable")
            continue
        e["new_title"] = title
        e["new_name"] = naming.build_output_filename(day, title)
    return plan


# --- stage 3: apply -----------------------------------------------------------

def rewrite_title(text: str, new_title: str, has_frontmatter: bool = True) -> str:
    """Replace the frontmatter `Title:` line (if any) and the first `# ` heading."""
    if has_frontmatter:
        text = re.sub(
            r"^Title: .*$", lambda _m: f"Title: {yaml_escape(new_title)}", text, count=1, flags=re.M
        )
    text = re.sub(r"^# .*$", lambda _m: f"# {new_title}", text, count=1, flags=re.M)
    return text


def apply_entry(entry: dict, folder: Path, pretty_subdir: str, log=print) -> bool:
    new_name = entry.get("new_name")
    new_title = entry.get("new_title")
    if not new_name or not new_title:
        return False

    old = folder / entry["file"]
    if not old.exists():
        log(f"skip {entry['file']}: not found in {folder}")
        return False

    old.write_text(
        rewrite_title(old.read_text(encoding="utf-8"), new_title, has_frontmatter=True),
        encoding="utf-8",
    )
    dst = naming.resolve_collision(folder, new_name)  # reserves the path atomically
    os.replace(old, dst)
    log(f"{entry['file']} -> {dst.name}")

    pretty = folder / pretty_subdir / entry["file"]
    if pretty.exists():
        pretty.write_text(
            rewrite_title(pretty.read_text(encoding="utf-8"), new_title, has_frontmatter=False),
            encoding="utf-8",
        )
        pdst = naming.resolve_collision(folder / pretty_subdir, dst.name)
        os.replace(pretty, pdst)
        log(f"  pretty: {entry['file']} -> {pdst.name}")
    return True


def apply_plan(plan: dict, folder: Path, pretty_subdir: str, log=print) -> int:
    applied = 0
    for entry in plan.get("files", []):
        if entry.get("action") != "rename":
            continue
        if apply_entry(entry, folder, pretty_subdir, log):
            applied += 1
    return applied


# --- CLI ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="transcriber.rename", description="LLM-driven rename pass over generated .md docs"
    )
    parser.add_argument("--folder", default=DEFAULT_FOLDER, help="folder of generated .md docs")
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--pretty-subdir", dest="pretty_subdir", default=DEFAULT_PRETTY_SUBDIR)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model", default=None, help="Ollama model (defaults to config llm_model)")
    parser.add_argument("--config", default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--classify", action="store_true")
    group.add_argument("--propose", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    log = logging.getLogger("rename")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    folder = Path(args.folder)
    plan_path = Path(args.plan)
    model = args.model or load_config(args.config).llm_model

    if args.classify:
        plan = build_classify_plan(folder, model, log, args.batch_size)
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        n = sum(1 for f in plan["files"] if f["action"] == "rename")
        print(f"Wrote {plan_path} — {n}/{len(plan['files'])} file(s) flagged for renaming.")
        print("Review/edit actions, then run --propose.")
        return 0

    if args.propose:
        if not plan_path.exists():
            print(f"Plan not found: {plan_path}. Run --classify first.")
            return 1
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        fill_proposals(plan, folder, model, log, args.batch_size)
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        n = sum(1 for f in plan["files"] if f.get("new_name"))
        print(f"Updated {plan_path} — {n} proposed name(s).")
        print("Review/edit new_name/new_title, then run --apply.")
        return 0

    # --apply
    if not plan_path.exists():
        print(f"Plan not found: {plan_path}. Run --classify then --propose first.")
        return 1
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    applied = apply_plan(plan, folder, args.pretty_subdir)
    print(f"Applied {applied} rename(s) in {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
