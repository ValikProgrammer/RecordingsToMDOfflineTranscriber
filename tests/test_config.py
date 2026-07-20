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
