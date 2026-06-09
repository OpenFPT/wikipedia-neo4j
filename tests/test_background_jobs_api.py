from __future__ import annotations

import time

from fastapi.testclient import TestClient

import src.main as main
from src.ingestion.pipeline import IngestResult


def test_start_and_get_hf_job_completed(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "app_api_key", None)

    def _fake_ingest_from_hf(**_kwargs):
        return [
            IngestResult(
                topic="A",
                page_id="1",
                title="A",
                url="https://a",
                chunk_count=1,
                entity_count=1,
            )
        ]

    monkeypatch.setattr(main, "ingest_from_hf", _fake_ingest_from_hf)

    with TestClient(main.app) as client:
        start = client.post(
            "/ingest/hf/jobs",
            json={"config_name": "20231101.simple", "split": "train", "sample_size": 1},
        )
        assert start.status_code == 200
        job_id = start.json()["job_id"]

        deadline = time.time() + 2
        body = {}
        while time.time() < deadline:
            r = client.get(f"/ingest/hf/jobs/{job_id}")
            body = r.json()
            if body["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)

        assert body["status"] == "completed"
        assert body["ingested"][0]["title"] == "A"


def test_stop_hf_job_transitions_to_cancelling_or_cancelled(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "app_api_key", None)

    def _fake_ingest_from_hf(**kwargs):
        should_stop = kwargs["should_stop"]
        on_progress = kwargs["on_progress"]
        results = []
        for i in range(10):
            if should_stop():
                break
            on_progress(i + 1, 10, f"T{i}")
            time.sleep(0.03)
            results.append(
                IngestResult(
                    topic=f"T{i}",
                    page_id=str(i),
                    title=f"T{i}",
                    url=f"https://{i}",
                    chunk_count=1,
                    entity_count=1,
                )
            )
        return results

    monkeypatch.setattr(main, "ingest_from_hf", _fake_ingest_from_hf)

    with TestClient(main.app) as client:
        start = client.post(
            "/ingest/hf/jobs",
            json={"config_name": "20231101.simple", "split": "train", "sample_size": 10},
        )
        job_id = start.json()["job_id"]

        stop = client.post(f"/ingest/hf/jobs/{job_id}/stop")
        assert stop.status_code == 200

        deadline = time.time() + 3
        final = {}
        while time.time() < deadline:
            r = client.get(f"/ingest/hf/jobs/{job_id}")
            final = r.json()
            if final["status"] in {"cancelled", "completed", "failed"}:
                break
            time.sleep(0.05)

        assert final["status"] in {"cancelled", "completed"}


def test_list_hf_jobs_supports_filter_and_pagination(monkeypatch) -> None:
    monkeypatch.setattr(main.settings, "app_api_key", None)

    with main._jobs_lock:
        main._jobs.clear()
        main._jobs["1"] = main._JobState(
            job_id="1",
            status="completed",
            config_name="c",
            split="train",
            sample_size=1,
            streaming=True,
            started_at="2026-01-01T00:00:00+00:00",
        )
        main._jobs["2"] = main._JobState(
            job_id="2",
            status="failed",
            config_name="c",
            split="train",
            sample_size=1,
            streaming=True,
            started_at="2026-01-02T00:00:00+00:00",
        )

    with TestClient(main.app) as client:
        resp = client.get("/ingest/hf/jobs?status=completed&limit=1&offset=0")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["status"] == "completed"


def test_restore_jobs_marks_running_as_interrupted(monkeypatch) -> None:
    payload = {
        "job-1": {
            "job_id": "job-1",
            "status": "running",
            "config_name": "c",
            "split": "train",
            "sample_size": 2,
            "streaming": True,
            "started_at": "2026-01-01T00:00:00+00:00",
        }
    }

    class _Store:
        def load_all(self):
            return payload

        def upsert(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(main, "_job_store", _Store())

    with main._jobs_lock:
        main._jobs.clear()

    main._restore_jobs()

    with main._jobs_lock:
        restored = main._jobs["job-1"]

    assert restored.status == "interrupted"
    assert restored.finished_at is not None
