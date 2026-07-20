from transcriber.manifest import Manifest
from transcriber.models import ManifestEntry
from transcriber.stages.ingest import compute_blake2b, scan_and_hash, scan_audio_files


def test_scan_finds_only_audio_extensions(tmp_path):
    (tmp_path / "a.m4a").write_bytes(b"1")
    (tmp_path / "b.mp3").write_bytes(b"2")
    (tmp_path / "notes.txt").write_bytes(b"3")
    (tmp_path / "clip.mov").write_bytes(b"4")
    found = {p.name for p in scan_audio_files(tmp_path)}
    assert found == {"a.m4a", "b.mp3", "clip.mov"}


def test_hash_is_stable_for_same_content(tmp_path):
    path = tmp_path / "a.m4a"
    path.write_bytes(b"same content")
    assert compute_blake2b(path) == compute_blake2b(path)


def test_hash_differs_for_different_content(tmp_path):
    p1 = tmp_path / "a.m4a"
    p2 = tmp_path / "b.m4a"
    p1.write_bytes(b"content one")
    p2.write_bytes(b"content two")
    assert compute_blake2b(p1) != compute_blake2b(p2)


def test_new_file_is_to_do(tmp_path):
    (tmp_path / "a.m4a").write_bytes(b"data")
    manifest = Manifest(tmp_path / "manifest.json")
    tasks = scan_and_hash(tmp_path, manifest)
    assert len(tasks) == 1
    assert tasks[0].status == "to_do"


def test_rename_does_not_retrigger_processing(tmp_path):
    path = tmp_path / "a.m4a"
    path.write_bytes(b"data")
    manifest = Manifest(tmp_path / "manifest.json")
    content_hash = compute_blake2b(path)
    manifest.upsert(ManifestEntry(content_hash=content_hash, source_name="a.m4a", status="done"))

    path.rename(tmp_path / "renamed.m4a")
    tasks = scan_and_hash(tmp_path, manifest)
    assert len(tasks) == 1
    assert tasks[0].status == "skip"


def test_content_change_triggers_redo(tmp_path):
    path = tmp_path / "a.m4a"
    path.write_bytes(b"data")
    manifest = Manifest(tmp_path / "manifest.json")
    old_hash = compute_blake2b(path)
    manifest.upsert(ManifestEntry(content_hash=old_hash, source_name="a.m4a", status="done"))

    path.write_bytes(b"different data now")
    tasks = scan_and_hash(tmp_path, manifest)
    assert len(tasks) == 1
    assert tasks[0].status == "to_do"
    assert tasks[0].content_hash != old_hash


def test_in_progress_is_redone_on_restart(tmp_path):
    path = tmp_path / "a.m4a"
    path.write_bytes(b"data")
    manifest = Manifest(tmp_path / "manifest.json")
    content_hash = compute_blake2b(path)
    manifest.upsert(ManifestEntry(content_hash=content_hash, source_name="a.m4a", status="in_progress"))

    tasks = scan_and_hash(tmp_path, manifest)
    assert tasks[0].status == "redo"


def test_failed_is_skipped_without_retry_flag(tmp_path):
    path = tmp_path / "a.m4a"
    path.write_bytes(b"data")
    manifest = Manifest(tmp_path / "manifest.json")
    content_hash = compute_blake2b(path)
    manifest.upsert(ManifestEntry(content_hash=content_hash, source_name="a.m4a", status="failed", error="boom"))

    tasks = scan_and_hash(tmp_path, manifest, retry_failed=False)
    assert tasks[0].status == "skip"


def test_failed_is_redone_with_retry_flag(tmp_path):
    path = tmp_path / "a.m4a"
    path.write_bytes(b"data")
    manifest = Manifest(tmp_path / "manifest.json")
    content_hash = compute_blake2b(path)
    manifest.upsert(ManifestEntry(content_hash=content_hash, source_name="a.m4a", status="failed", error="boom"))

    tasks = scan_and_hash(tmp_path, manifest, retry_failed=True)
    assert tasks[0].status == "redo"
