from transcriber.stages.langdetect import decide_language, window_starts


def test_window_starts_short_file_single_window():
    assert window_starts(20.0, 3) == [0.0]
    assert window_starts(30.0, 3) == [0.0]


def test_window_starts_spreads_and_stays_in_range():
    starts = window_starts(600.0, 3)  # 10 min
    assert len(starts) == 3
    assert starts[0] >= 0.0
    assert starts == sorted(starts)
    assert all(s + 30 <= 600.0 + 1e-6 for s in starts)  # each 30s window fits


def test_window_starts_dedupes_near_identical():
    # a file only a bit longer than one window can't fit 3 distinct windows
    starts = window_starts(35.0, 3)
    assert starts == sorted(set(starts))


def test_decide_language_all_agree_forces():
    assert decide_language([("ru", 0.9), ("ru", 0.8), ("ru", 0.95)], min_prob=0.6) == "ru"


def test_decide_language_disagreement_returns_none():
    # bilingual: confident windows disagree -> don't force, let backend switch
    assert decide_language([("ru", 0.9), ("en", 0.9)], min_prob=0.6) is None


def test_decide_language_drops_low_confidence_noise():
    # a noisy window guesses "cy" but below threshold -> ignored; the rest agree
    assert decide_language([("ru", 0.9), ("cy", 0.2), ("ru", 0.85)], min_prob=0.6) == "ru"


def test_decide_language_all_noise_returns_none():
    assert decide_language([("cy", 0.2), ("nn", 0.3), (None)], min_prob=0.6) is None


def test_decide_language_low_conf_agreement_still_none():
    # nothing clears the bar -> auto (don't force on a guess)
    assert decide_language([("ru", 0.4), ("ru", 0.5)], min_prob=0.6) is None
