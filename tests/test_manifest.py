import threading

from transcriber.manifest import Manifest
from transcriber.models import ManifestEntry


def _entry(content_hash="blake2b:aaa", status="done", source_name="a.m4a", **kwargs) -> ManifestEntry:
    return ManifestEntry(content_hash=content_hash, source_name=source_name, status=status, **kwargs)


def test_load_missing_file_returns_empty(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    assert manifest.all_entries() == {}
    assert manifest.get("blake2b:aaa") is None


def test_atomic_save_leaves_no_tmp_file(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    manifest.upsert(_entry())
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()


def test_upsert_persists_and_reloads(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    manifest.upsert(_entry(out_path="out/x.md"))

    reloaded = Manifest(path)
    entry = reloaded.get("blake2b:aaa")
    assert entry is not None
    assert entry.out_path == "out/x.md"
    assert entry.status == "done"


def test_dedup_by_hash_same_hash_overwrites(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.upsert(_entry(status="in_progress"))
    manifest.upsert(_entry(status="done", out_path="out/x.md"))
    assert len(manifest.all_entries()) == 1
    assert manifest.get("blake2b:aaa").status == "done"


def test_rename_keeps_same_hash_entry(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.upsert(_entry(source_name="original.m4a"))
    manifest.upsert(_entry(source_name="renamed.m4a"))
    entries = manifest.all_entries()
    assert len(entries) == 1
    assert entries["blake2b:aaa"].source_name == "renamed.m4a"


def test_entries_with_status_filters_correctly(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.upsert(_entry(content_hash="blake2b:aaa", status="done"))
    manifest.upsert(_entry(content_hash="blake2b:bbb", status="failed"))
    manifest.upsert(_entry(content_hash="blake2b:ccc", status="in_progress"))

    assert {e.content_hash for e in manifest.entries_with_status("failed")} == {"blake2b:bbb"}
    assert {e.content_hash for e in manifest.entries_with_status("in_progress")} == {"blake2b:ccc"}


def test_concurrent_upserts_are_thread_safe(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")

    def worker(i):
        manifest.upsert(_entry(content_hash=f"blake2b:h{i}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(manifest.all_entries()) == 20
    reloaded = Manifest(tmp_path / "manifest.json")
    assert len(reloaded.all_entries()) == 20


def test_migrate_legacy_done_sets_only_text_done(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"schema": 1, "entries": {"blake2b:aaa": {'
        '"content_hash": "blake2b:aaa", "source_name": "a.m4a", "status": "done",'
        '"updated_at": "2026-01-01T00:00:00Z"}}}'
    )
    from transcriber.manifest import Manifest
    m = Manifest(path)
    e = m.get("blake2b:aaa")
    assert e.stages["text"].status == "done"
    assert e.stages["diarize"].status == "pending"
    assert e.stages["summary"].status == "pending"
    assert e.stages["pretty"].status == "pending"


def test_migrate_does_not_infer_summary_from_elsewhere(tmp_path):
    # even if we later have raw with summary, migration alone must leave summary pending
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"schema": 1, "entries": {"blake2b:aaa": {'
        '"content_hash": "blake2b:aaa", "source_name": "a.m4a", "status": "done"}}}'
    )
    from transcriber.manifest import Manifest
    e = Manifest(path).get("blake2b:aaa")
    assert e.stages["summary"].status == "pending"


def test_new_entry_has_all_stages_pending(tmp_path):
    from transcriber.manifest import Manifest
    from transcriber.models import ManifestEntry, default_stages
    m = Manifest(tmp_path / "manifest.json")
    m.upsert(ManifestEntry(content_hash="blake2b:x", source_name="x.m4a", status="in_progress", stages=default_stages()))
    e = Manifest(tmp_path / "manifest.json").get("blake2b:x")
    assert all(e.stages[s].status == "pending" for s in ("text", "diarize", "summary", "pretty"))
