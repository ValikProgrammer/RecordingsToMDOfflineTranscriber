from transcriber.progress import NullReporter, ProgressReporter, RateModel, SlotPool, fmt_duration


# --- fmt_duration --------------------------------------------------------

def test_fmt_duration_under_an_hour_is_m_ss():
    assert fmt_duration(0) == "0:00"
    assert fmt_duration(5) == "0:05"
    assert fmt_duration(72) == "1:12"


def test_fmt_duration_over_an_hour_is_h_mm_ss():
    assert fmt_duration(3661) == "1:01:01"


def test_fmt_duration_clamps_negative_to_zero():
    assert fmt_duration(-10) == "0:00"


# --- RateModel -----------------------------------------------------------

def test_rate_model_uses_default_rtf_until_a_file_completes():
    rm = RateModel(default_rtf=0.6, total_audio_sec=1000)
    assert rm.per_file_rtf == 0.6
    assert rm.est_wall(100) == 60.0


def test_rate_model_learns_rtf_from_completed_files():
    rm = RateModel(default_rtf=0.6, total_audio_sec=1000)
    rm.record(audio_sec=100, wall_sec=50)  # rtf 0.5
    assert rm.per_file_rtf == 0.5
    assert rm.est_wall(200) == 100.0
    assert rm.completed_audio == 100
    assert rm.completed_files == 1


def test_rate_model_batch_eta_none_before_first_completion():
    rm = RateModel(default_rtf=0.6, total_audio_sec=1000)
    assert rm.batch_eta(elapsed=10) is None


def test_rate_model_batch_eta_uses_throughput():
    rm = RateModel(default_rtf=0.6, total_audio_sec=100)
    rm.record(audio_sec=50, wall_sec=40)
    # throughput = 50 audio / 100 wall elapsed = 0.5; remaining 50 audio -> 100 wall
    assert rm.batch_eta(elapsed=100) == 100.0


def test_rate_model_batch_eta_none_when_finished():
    rm = RateModel(default_rtf=0.6, total_audio_sec=50)
    rm.record(audio_sec=50, wall_sec=40)
    assert rm.batch_eta(elapsed=40) is None


def test_rate_model_est_wall_floors_at_one_second():
    rm = RateModel(default_rtf=0.6, total_audio_sec=0)
    assert rm.est_wall(0) == 1.0


# --- SlotPool ------------------------------------------------------------

def test_slot_pool_hands_out_smallest_free_row_starting_at_one():
    pool = SlotPool()
    assert pool.acquire() == 1
    assert pool.acquire() == 2
    pool.release(1)
    assert pool.acquire() == 1  # reuses the freed row
    assert pool.acquire() == 3


# --- reporters (no TTY / no bars) ----------------------------------------

def test_null_reporter_is_fully_no_op():
    r = NullReporter()
    r.start_batch(100, 2)
    r.file_start("h", 1, 2, "a.m4a", 50)
    r.stage("h", "WHISPER", "transcribing")
    r.file_done("h", 30, "/out/a.md")
    r.file_failed("h", RuntimeError("x"))
    r.close()  # must not raise


def test_progress_reporter_without_bars_runs_full_lifecycle():
    # bars=False -> prints tagged lines, no tqdm bars, no ticker thread
    r = ProgressReporter(default_rtf=0.6, bars=False)
    r.start_batch(100, 1)
    r.file_start("h", 1, 1, "a.m4a", 50)
    r.stage("h", "WHISPER", "transcribing")
    r.file_done("h", 30, "/out/a.md")
    r.close()
    assert r._rate.completed_files == 1
    assert r._rate.completed_audio == 50


def test_progress_reporter_file_failed_without_prior_start_does_not_raise():
    r = ProgressReporter(default_rtf=0.6, bars=False)
    r.start_batch(0, 0)
    r.file_failed("unknown", RuntimeError("boom"))
    r.close()
