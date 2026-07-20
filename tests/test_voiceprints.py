from transcriber.voiceprints import VoiceprintStore


def test_store_creates_directory(tmp_path):
    store_dir = tmp_path / "voiceprints"
    VoiceprintStore(store_dir)
    assert store_dir.exists()


def test_enroll_does_not_raise(tmp_path):
    store = VoiceprintStore(tmp_path / "voiceprints")
    store.enroll("Jamie", [0.1, 0.2, 0.3])


def test_identify_returns_none_in_v1(tmp_path):
    store = VoiceprintStore(tmp_path / "voiceprints")
    store.enroll("Jamie", [0.1, 0.2, 0.3])
    assert store.identify([0.1, 0.2, 0.3]) is None
