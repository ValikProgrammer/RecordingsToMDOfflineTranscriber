from transcriber.cli import apply_overrides, build_run_options, parse_args, parse_names, resolve_mode
from transcriber.config import Config


def test_input_folder_synonyms_all_set_same_dest():
    for flag in ("--input-folder", "--folder", "--input"):
        args = parse_args([flag, "./audio"])
        assert args.input_folder == "./audio"


def test_skip_and_exclude_are_synonyms():
    assert parse_args(["--skip", "a.m4a", "b.m4a"]).skip == ["a.m4a", "b.m4a"]
    assert parse_args(["--exclude", "a.m4a"]).skip == ["a.m4a"]


def test_transcribe_and_text_are_synonyms_for_text_mode():
    assert resolve_mode(parse_args(["--transcribe"])) == "text"
    assert resolve_mode(parse_args(["--text"])) == "text"


def test_resolve_mode_defaults_to_full():
    assert resolve_mode(parse_args([])) == "full"


def test_resolve_mode_priority_summary_resummarize_rerender():
    assert resolve_mode(parse_args(["--summary"])) == "summary"
    assert resolve_mode(parse_args(["--resummarize"])) == "resummarize"
    assert resolve_mode(parse_args(["--rerender"])) == "rerender"


def test_resolve_mode_diarize():
    assert resolve_mode(parse_args(["--diarize"])) == "diarize"


def test_diarize_alone_is_postpass_mode():
    assert resolve_mode(parse_args(["--diarize"])) == "diarize"


def test_text_plus_diarize_sets_want_diarize():
    args = parse_args(["--text", "--diarize"])
    assert resolve_mode(args) == "text"
    opts = build_run_options(args, "text")
    assert opts.want_diarize is True


def test_text_alone_leaves_want_diarize_false():
    args = parse_args(["--text"])
    assert resolve_mode(args) == "text"
    opts = build_run_options(args, "text")
    assert opts.want_diarize is False


def test_full_wants_diarize():
    opts = build_run_options(parse_args([]), "full")
    assert opts.want_diarize is True


def test_diarize_mode_wants_diarize():
    args = parse_args(["--diarize"])
    opts = build_run_options(args, "diarize")
    assert opts.want_diarize is True


def test_parse_names_splits_and_trims():
    assert parse_names("Alex, Jamie") == ["Alex", "Jamie"]
    assert parse_names(None) is None
    assert parse_names("") is None


def test_apply_overrides_sets_config_fields():
    cfg = Config()
    args = parse_args([
        "--input-folder", "./a", "--out", "./b", "--llm-model", "custom:model",
        "--llm-ctx", "40000",
        "--jobs", "5", "--diarize-device", "cpu", "--no-frontmatter", "--wikilink-speakers",
    ])
    cfg = apply_overrides(cfg, args)
    assert cfg.input_folder == "./a"
    assert cfg.out_folder == "./b"
    assert cfg.llm_model == "custom:model"
    assert cfg.llm_ctx == 40000
    assert cfg.jobs == 5
    assert cfg.diarize_device == "cpu"
    assert cfg.obsidian_frontmatter is False
    assert cfg.wikilink_speakers is True


def test_apply_overrides_leaves_defaults_when_flags_absent():
    cfg = apply_overrides(Config(), parse_args([]))
    assert cfg == Config()


def test_apply_overrides_sets_systems_folder_and_backend():
    cfg = apply_overrides(
        Config(), parse_args([
            "--systems-folder", "./cmp/cpu/systems", "--logs-folder", "./cmp/cpu/logs",
            "--backend", "faster-whisper",
        ])
    )
    assert cfg.systems_folder == "./cmp/cpu/systems"
    assert cfg.logs_folder == "./cmp/cpu/logs"
    assert cfg.asr_backend == "faster-whisper"


def test_beam_defaults_to_5():
    assert parse_args([]).beam == 5
    assert parse_args(["--beam", "1"]).beam == 1


def test_select_transcribe_picks_backend():
    from functools import partial

    from transcriber.__main__ import _select_transcribe
    from transcriber.stages import asr_mlx

    mlx_fn, mlx_msg = _select_transcribe(Config(asr_backend="mlx"), parse_args([]))
    assert mlx_fn is asr_mlx.transcribe
    assert "mlx" in mlx_msg

    fw_fn, fw_msg = _select_transcribe(Config(asr_backend="faster-whisper"), parse_args(["--beam", "3"]))
    assert isinstance(fw_fn, partial)
    assert fw_fn.keywords["beam_size"] == 3
    assert "faster-whisper" in fw_msg


def test_build_run_options_maps_flags():
    args = parse_args([
        "--only", "a", "--skip", "b.m4a", "--turbo", "--speakers", "2",
        "--min-speakers", "1", "--max-speakers", "3", "--names", "Alex,Jamie",
        "--wikilink-speakers",
    ])
    opts = build_run_options(args, "full")
    assert opts.mode == "full"
    assert opts.only == "a"
    assert opts.skip == ["b.m4a"]
    assert opts.turbo is True
    assert opts.speakers == 2
    assert opts.min_speakers == 1
    assert opts.max_speakers == 3
    assert opts.names == ["Alex", "Jamie"]
    assert opts.wikilink_speakers is True
    assert opts.frontmatter is True


def test_build_run_options_no_frontmatter_flag():
    args = parse_args(["--no-frontmatter"])
    opts = build_run_options(args, "full")
    assert opts.frontmatter is False


def test_language_flag_overrides_asr_language():
    from transcriber.cli import apply_overrides, parse_args
    from transcriber.config import Config

    cfg = apply_overrides(Config(), parse_args(["--language", "en"]))
    assert cfg.asr_language == "en"


def test_no_language_flag_keeps_config_asr_language():
    from transcriber.cli import apply_overrides, parse_args
    from transcriber.config import Config

    cfg = Config(asr_language="ru")
    cfg = apply_overrides(cfg, parse_args([]))
    assert cfg.asr_language == "ru"


def test_enroll_flag_parsed():
    from transcriber.cli import parse_args

    assert parse_args(["--enroll", "Галя"]).enroll == "Галя"
    assert parse_args([]).enroll is None


def test_force_flag_parsed_and_mapped_to_run_option():
    from transcriber.cli import build_run_options, parse_args

    assert parse_args(["--summary", "--force"]).force is True
    assert parse_args(["--summary"]).force is False
    assert build_run_options(parse_args(["--summary", "--force"]), "summary").force is True
    assert build_run_options(parse_args(["--summary"]), "summary").force is False


def test_pretty_flag_sets_run_option():
    from transcriber.cli import build_run_options, parse_args, resolve_mode

    args = parse_args(["--pretty"])
    opts = build_run_options(args, resolve_mode(args))
    assert opts.pretty is True
    assert build_run_options(parse_args([]), "full").pretty is False
