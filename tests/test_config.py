from transcriber.config import Config, load_config


def test_defaults_when_no_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("transcriber.config.USER_CONFIG_PATH", tmp_path / "nonexistent.toml")
    cfg = load_config()
    assert cfg == Config()


def test_load_explicit_path_overrides_simple_keys(tmp_path):
    toml_path = tmp_path / "custom.toml"
    toml_path.write_text(
        """
input_folder = "./my_audio"
jobs = 7
diarize_device = "cpu"
obsidian_frontmatter = false
"""
    )
    cfg = load_config(str(toml_path))
    assert cfg.input_folder == "./my_audio"
    assert cfg.jobs == 7
    assert cfg.diarize_device == "cpu"
    assert cfg.obsidian_frontmatter is False
    # untouched keys keep their defaults
    assert cfg.llm_model == "qwen2.5:14b"


def test_load_explicit_path_missing_file_falls_back_to_defaults(tmp_path):
    cfg = load_config(str(tmp_path / "does-not-exist.toml"))
    assert cfg == Config()


def test_tiers_override_from_toml(tmp_path):
    toml_path = tmp_path / "custom.toml"
    toml_path.write_text(
        """
[summary]
tiers = [
  { up_to_min = 10, sentences = "2-3" },
  { up_to_min = 100000, sentences = "6-9" },
]
long_form_from_min = 30
"""
    )
    cfg = load_config(str(toml_path))
    assert cfg.summary_tiers[0].up_to_min == 10
    assert cfg.summary_tiers[0].sentences == "2-3"
    assert len(cfg.summary_tiers) == 2
    assert cfg.long_form_from_min == 30


def test_cwd_config_toml_is_picked_up_when_no_explicit_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("transcriber.config.USER_CONFIG_PATH", tmp_path / "nonexistent.toml")
    (tmp_path / "config.toml").write_text('jobs = 9\n')
    cfg = load_config()
    assert cfg.jobs == 9


def test_load_config_reads_asr_language_and_prompt_extra(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('asr_language = "en"\nasr_prompt_extra = "ФизТех, Богодаров"\n')
    monkeypatch.chdir(tmp_path)

    cfg = load_config(None)

    assert cfg.asr_language == "en"
    assert cfg.asr_prompt_extra == "ФизТех, Богодаров"


def test_config_defaults_asr_language_ru():
    assert Config().asr_language == "ru"
    assert Config().asr_prompt_extra == ""


def test_parse_prompt_file_one_term_per_line_ignores_blanks_and_comments():
    from transcriber.config import parse_prompt_file

    text = "# header\nГрузия\n\nМиша  \n  # a comment\nfast track # inline note\n"
    assert parse_prompt_file(text) == ["Грузия", "Миша", "fast track"]


def test_load_config_merges_prompt_file_into_extra(tmp_path, monkeypatch):
    (tmp_path / "glossary.local.txt").write_text("Грузия\nМиша\n")
    (tmp_path / "config.toml").write_text(
        'asr_prompt_extra = "ФизТех"\nasr_prompt_file = "./glossary.local.txt"\n'
    )
    monkeypatch.chdir(tmp_path)

    cfg = load_config(None)

    assert cfg.asr_prompt_extra == "ФизТех, Грузия, Миша"


def test_load_config_prompt_file_alone_when_no_inline(tmp_path, monkeypatch):
    (tmp_path / "g.txt").write_text("Грузия\nМиша\n")
    (tmp_path / "config.toml").write_text('asr_prompt_file = "./g.txt"\n')
    monkeypatch.chdir(tmp_path)

    cfg = load_config(None)

    assert cfg.asr_prompt_extra == "Грузия, Миша"


def test_load_config_missing_prompt_file_is_noop(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text('asr_prompt_file = "./nope.txt"\n')
    monkeypatch.chdir(tmp_path)

    cfg = load_config(None)

    assert cfg.asr_prompt_extra == ""
