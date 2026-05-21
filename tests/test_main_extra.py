"""Additional tests for main.py — rate limiter, ingest endpoints, job management."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

import src.main as main
from src.ingest import IngestResult


class TestRateLimiter:
    def test_allows_within_limit(self) -> None:
        limiter = main._RateLimiter(max_requests=3, period_seconds=60)
        allowed, remaining = limiter.allow("client1")
        assert allowed is True
        assert remaining == 2

    def test_rejects_over_limit(self) -> None:
        limiter = main._RateLimiter(max_requests=2, period_seconds=60)
        limiter.allow("client1")
        limiter.allow("client1")
        allowed, remaining = limiter.allow("client1")
        assert allowed is False
        assert remaining == 0

    def test_separate_clients_have_separate_buckets(self) -> None:
        limiter = main._RateLimiter(max_requests=1, period_seconds=60)
        assert limiter.allow("a")[0] is True
        assert limiter.allow("b")[0] is True
        assert limiter.allow("a")[0] is False

    def test_window_expiry_resets_counter(self) -> None:
        limiter = main._RateLimiter(max_requests=1, period_seconds=1)
        assert limiter.allow("x")[0] is True
        assert limiter.allow("x")[0] is False
        time.sleep(1.1)
        assert limiter.allow("x")[0] is True


class TestIngestEndpoint:
    def test_ingest_success(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        def _fake_ingest_topic(topic):
            return IngestResult(
                topic=topic,
                page_id="p1",
                title=topic,
                url=f"https://en.wikipedia.org/wiki/{topic}",
                chunk_count=3,
                entity_count=5,
            )

        monkeypatch.setattr(main, "ingest_topic", _fake_ingest_topic)

        with TestClient(main.app) as client:
            resp = client.post("/ingest", json={"topics": ["Neo4j"]})

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["ingested"]) == 1
        assert body["ingested"][0]["title"] == "Neo4j"
        assert body["ingested"][0]["chunk_count"] == 3

    def test_ingest_value_error_returns_400(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        def _fail(topic):
            raise ValueError("Ambiguous topic")

        monkeypatch.setattr(main, "ingest_topic", _fail)

        with TestClient(main.app) as client:
            resp = client.post("/ingest", json={"topics": ["Python"]})

        assert resp.status_code == 400
        assert "Ambiguous" in resp.json()["detail"]

    def test_ingest_empty_topics_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        with TestClient(main.app) as client:
            resp = client.post("/ingest", json={"topics": []})

        assert resp.status_code == 422


class TestHFIngestEndpoint:
    def test_sync_hf_ingest_success(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        def _fake_ingest(**kwargs):
            return [
                IngestResult(topic="A", page_id="1", title="A", url="https://a", chunk_count=2, entity_count=3)
            ]

        monkeypatch.setattr(main, "ingest_from_hf", _fake_ingest)

        with TestClient(main.app) as client:
            resp = client.post("/ingest/hf", json={"config_name": "20231101.en", "sample_size": 1})

        assert resp.status_code == 200
        assert resp.json()["ingested"][0]["title"] == "A"

    def test_sync_hf_ingest_runtime_error_returns_400(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        def _fail(**kwargs):
            raise RuntimeError("dataset not found")

        monkeypatch.setattr(main, "ingest_from_hf", _fail)

        with TestClient(main.app) as client:
            resp = client.post("/ingest/hf", json={"config_name": "bad", "sample_size": 1})

        assert resp.status_code == 400


class TestJobNotFound:
    def test_get_nonexistent_job_returns_404(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        with TestClient(main.app) as client:
            resp = client.get("/ingest/hf/jobs/nonexistent-id")

        assert resp.status_code == 404

    def test_stop_nonexistent_job_returns_404(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        with TestClient(main.app) as client:
            resp = client.post("/ingest/hf/jobs/nonexistent-id/stop")

        assert resp.status_code == 404


class TestSerializeIngestResult:
    def test_serialization_shape(self) -> None:
        result = IngestResult(
            topic="Test",
            page_id="p1",
            title="Test Title",
            url="https://example.org",
            chunk_count=5,
            entity_count=10,
        )
        serialized = main._serialize_ingest_result(result)
        assert serialized == {
            "topic": "Test",
            "page_id": "p1",
            "title": "Test Title",
            "url": "https://example.org",
            "chunk_count": 5,
            "entity_count": 10,
        }


class TestQueryEndpoint:
    def test_query_runtime_error_returns_500(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        def _fail(question, top_k):
            raise RuntimeError("Neo4j down")

        monkeypatch.setattr(main, "query_graph", _fail)

        with TestClient(main.app) as client:
            resp = client.post("/query", json={"question": "What is Neo4j?", "top_k": 3})

        assert resp.status_code == 500
