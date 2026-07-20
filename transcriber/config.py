"""Config loading: config.toml (CWD, then ~/.config/transcriber/) + defaults (§6.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback per §3
    import tomli as tomllib  # type: ignore

DEFAULT_CONFIG_NAME = "config.toml"
USER_CONFIG_PATH = Path.home() / ".config" / "transcriber" / "config.toml"

_SIMPLE_KEYS = (
    "input_folder", "out_folder", "systems_folder", "logs_folder",
    "asr_model", "asr_language", "asr_prompt_extra", "asr_artifact_denylist_extra",
    "llm_model", "llm_ctx", "diarize_device",
    "mono_threshold", "jobs", "obsidian_frontmatter", "wikilink_speakers",
)


@dataclass
class SummaryTier:
    up_to_min: float
    sentences: str


def _default_tiers() -> list[SummaryTier]:
    return [
        SummaryTier(15, "3–5"),
        SummaryTier(45, "5–8"),
        SummaryTier(90, "8–12"),
        SummaryTier(100000, "10–15"),
    ]


@dataclass
class Config:
    input_folder: str = "./audio"
    out_folder: str = "./out"
    systems_folder: str = "./systems"
    logs_folder: str = "./logs"
    asr_model: str = "large-v3"
    asr_language: str = "ru"
    asr_prompt_extra: str = ""
    asr_artifact_denylist_extra: list[str] = field(default_factory=list)
    llm_model: str = "qwen2.5:14b"
    llm_ctx: int = 8192
    diarize_device: str = "mps"
    mono_threshold: float = 0.92
    jobs: int = 3
    obsidian_frontmatter: bool = True
    wikilink_speakers: bool = False
    summary_tiers: list[SummaryTier] = field(default_factory=_default_tiers)
    long_form_from_min: float = 45


def find_config_path(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    cwd_cfg = Path.cwd() / DEFAULT_CONFIG_NAME
    if cwd_cfg.exists():
        return cwd_cfg
    if USER_CONFIG_PATH.exists():
        return USER_CONFIG_PATH
    return None


def load_config(explicit_path: str | None = None) -> Config:
    cfg = Config()
    path = find_config_path(explicit_path)
    if path is None:
        return cfg
    with open(path, "rb") as f:
        data = tomllib.load(f)
    _apply_toml(cfg, data)
    return cfg


def _apply_toml(cfg: Config, data: dict) -> None:
    for key in _SIMPLE_KEYS:
        if key in data:
            setattr(cfg, key, data[key])
    summary = data.get("summary", {})
    if "tiers" in summary:
        cfg.summary_tiers = [
            SummaryTier(t["up_to_min"], t["sentences"]) for t in summary["tiers"]
        ]
    if "long_form_from_min" in summary:
        cfg.long_form_from_min = summary["long_form_from_min"]
