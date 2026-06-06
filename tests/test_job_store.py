from __future__ import annotations

from pathlib import Path

from src.infrastructure.job_store import JobStore


def test_upsert_persists_payload(tmp_path: Path) -> None:
    store_path = tmp_path / "jobs.json"
    store = JobStore(str(store_path))

    store.upsert("job-1", {"status": "running"})

    data = store.load_all()
    assert data["job-1"]["status"] == "running"


def test_load_all_recovers_from_corrupted_file(tmp_path: Path) -> None:
    store_path = tmp_path / "jobs.json"
    store_path.write_text("{not-json", encoding="utf-8")

    store = JobStore(str(store_path))

    assert store.load_all() == {}


def test_atomic_writes_leave_valid_json(tmp_path: Path) -> None:
    store_path = tmp_path / "jobs.json"
    store = JobStore(str(store_path))

    for i in range(10):
        store.upsert(f"job-{i}", {"status": "ok", "i": i})

    content = store_path.read_text(encoding="utf-8")
    assert content.strip().startswith("{")
    assert "job-9" in content
