"""Additional tests for main.py — rate limiter, ingest endpoints, job management."""

from __future__ import annotations

import time
from unittest.mock import patch

from fastapi.testclient import TestClient

import src.main as main
from src.retrieval import fusion
from src.retrieval import reranker as retrieval_reranker
from src.ingestion.pipeline import IngestResult


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

    def test_query_includes_trace_only_when_debug_true(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        def _fake_query_graph(question, top_k):
            return type(
                "R",
                (),
                {
                    "answer": "ok",
                    "citations": [],
                    "retrieval_tier": "generated",
                    "trace": {"steps": [{"kind": "retrieval", "name": "generated"}]},
                },
            )()

        monkeypatch.setattr(main, "query_graph", _fake_query_graph)

        with TestClient(main.app) as client:
            debug_false = client.post(
                "/query",
                json={"question": "What is Neo4j?", "top_k": 3, "debug": False},
            )
            debug_true = client.post(
                "/query",
                json={"question": "What is Neo4j?", "top_k": 3, "debug": True},
            )

        assert debug_false.status_code == 200
        assert "trace" not in debug_false.json()
        assert debug_true.status_code == 200
        assert debug_true.json()["trace"]["steps"][0]["name"] == "generated"

    def test_query_allows_tauri_origins_via_cors(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        def _fake_query_graph(question, top_k):
            return type("R", (), {"answer": "ok", "citations": [], "retrieval_tier": "generated"})()

        monkeypatch.setattr(main, "query_graph", _fake_query_graph)

        with TestClient(main.app) as client:
            preflight = client.options(
                "/query",
                headers={
                    "Origin": "http://tauri.localhost",
                    "Access-Control-Request-Method": "POST",
                },
            )
            resp = client.post(
                "/query",
                json={"question": "What is Neo4j?", "top_k": 3},
                headers={"Origin": "http://tauri.localhost"},
            )

        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "http://tauri.localhost"
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://tauri.localhost"


class TestHybridQueryEndpoint:
    def test_hybrid_query_success(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        def _fake_retrieve(question, top_k):
            return [
                {"chunk_id": "c1", "chunk_text": "result", "score": 0.9, "page_title": "P", "page_url": "u"}
            ]

        monkeypatch.setattr(main, "hybrid_retrieve", _fake_retrieve)

        with TestClient(main.app) as client:
            resp = client.post("/query/hybrid", json={"question": "test?", "top_k": 5})

        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1

    def test_hybrid_query_runtime_error(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        def _fail(question, top_k):
            raise RuntimeError("Neo4j connection lost")

        monkeypatch.setattr(main, "hybrid_retrieve", _fail)

        with TestClient(main.app) as client:
            resp = client.post("/query/hybrid", json={"question": "test?", "top_k": 3})

        assert resp.status_code == 500

    def test_hybrid_query_logs_signal_scores(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        def _fake_retrieve(question, top_k):
            return [
                {"chunk_id": "c1", "chunk_text": "r", "score": 0.9, "bm25_score": 0.5, "vector_score": 0.4},
                {"chunk_id": "c2", "chunk_text": "r", "score": 0.8, "graph_rank": 2},
            ]

        monkeypatch.setattr(main, "hybrid_retrieve", _fake_retrieve)

        with TestClient(main.app) as client:
            resp = client.post("/query/hybrid", json={"question": "test?", "top_k": 5})

        assert resp.status_code == 200


class TestChatEndpoint:
    def test_chat_uses_same_query_pipeline_in_local_mode(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)
        monkeypatch.setattr(main.settings, "model_mode", "local")

        def _fake_query_graph(question: str, top_k: int):
            assert question == "Hồ Chí Minh là ai?"
            assert top_k == 2
            return type(
                "R",
                (),
                {
                    "answer": "Hồ Chí Minh là lãnh tụ cách mạng Việt Nam.",
                    "citations": [{"page_title": "Hồ Chí Minh", "page_url": "u", "chunk_id": "c1"}],
                    "retrieval_tier": "agent",
                },
            )()

        monkeypatch.setattr(main, "query_graph", _fake_query_graph)

        with TestClient(main.app) as client:
            resp = client.post(
                "/chat",
                json={
                    "messages": [{"role": "user", "content": "Hồ Chí Minh là ai?"}],
                    "top_k": 2,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "Hồ Chí Minh là lãnh tụ cách mạng Việt Nam." in data["answer"]
        assert data["citations"][0]["page_title"] == "Hồ Chí Minh"
        assert data["retrieval_tier"] == "agent"

    def test_chat_uses_same_query_pipeline_in_api_mode(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)
        monkeypatch.setattr(main.settings, "model_mode", "api")

        def _fake_query_graph(question: str, top_k: int):
            assert question == "Neo4j được sử dụng như thế nào?"
            assert top_k == 3
            return type(
                "R",
                (),
                {
                    "answer": "Neo4j được sử dụng để mô hình hóa và truy vấn dữ liệu dạng đồ thị.",
                    "citations": [{"page_title": "Neo4j", "page_url": "u", "chunk_id": "c1"}],
                    "retrieval_tier": "generated",
                },
            )()

        monkeypatch.setattr(main, "query_graph", _fake_query_graph)

        with TestClient(main.app) as client:
            resp = client.post(
                "/chat",
                json={
                    "messages": [{"role": "user", "content": "Neo4j được sử dụng như thế nào?"}],
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "mô hình hóa" in data["answer"]
        assert data["retrieval_tier"] == "generated"

    def test_chat_forwards_trace_only_when_debug_true(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        def _fake_query_graph(question: str, top_k: int):
            return type(
                "R",
                (),
                {
                    "answer": "A",
                    "citations": [],
                    "retrieval_tier": "generated",
                    "trace": {"steps": [{"kind": "retrieval", "name": "generated"}]},
                },
            )()

        monkeypatch.setattr(main, "query_graph", _fake_query_graph)

        with TestClient(main.app) as client:
            debug_false = client.post(
                "/chat",
                json={
                    "messages": [{"role": "user", "content": "Q??"}],
                    "top_k": 2,
                    "debug": False,
                },
            )
            debug_true = client.post(
                "/chat",
                json={
                    "messages": [{"role": "user", "content": "Q??"}],
                    "top_k": 2,
                    "debug": True,
                },
            )

        assert debug_false.status_code == 200
        assert "trace" not in debug_false.json()
        assert debug_true.status_code == 200
        assert debug_true.json()["trace"]["steps"][0]["name"] == "generated"

    def test_chat_stream_emits_live_events(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        def _fake_query_graph(question: str, top_k: int, emit=None):
            assert question == "Who?"
            assert top_k == 2
            assert emit is not None
            emit("route", {"route": "local_agent"})
            emit("tool_call", {"tool": "kg_query"})
            emit("cypher", {"query": "MATCH (n) RETURN n LIMIT 1"})
            return type(
                "R",
                (),
                {
                    "answer": "ok",
                    "citations": [],
                    "retrieval_tier": "generated",
                    "trace": {"tier": "generated", "steps": []},
                },
            )()

        monkeypatch.setattr(main, "query_graph", _fake_query_graph)

        with TestClient(main.app) as client:
            resp = client.post(
                "/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "Who?"}],
                    "top_k": 2,
                    "debug": True,
                },
            )

        assert resp.status_code == 200
        assert "event: route" in resp.text
        assert "event: tool_call" in resp.text
        assert "MATCH (n) RETURN n LIMIT 1" in resp.text
        assert "event: final" in resp.text


class TestFusionTrace:
    def test_synthesize_answer_abstains_when_llm_fails(self, monkeypatch) -> None:
        monkeypatch.setattr(
            fusion,
            "_client_pool",
            lambda: (_ for _ in ()).throw(RuntimeError("llm down")),
        )

        answer = fusion._synthesize_answer(
            "Hội nghị Genève về Đông Dương có tính chất như thế nào?",
            ["Đoạn 1", "Đoạn 2"],
        )

        assert "Dựa trên thông tin tìm được:" not in answer
        assert "Không đủ thông tin" in answer or "Không tìm thấy" in answer

    def test_query_graph_sets_generated_trace(self, monkeypatch) -> None:
        monkeypatch.setattr(fusion.settings, "model_mode", "api")
        monkeypatch.setattr(
            fusion,
            "_run_generated_query",
            lambda question, top_k: [
                {
                    "page_title": "P",
                    "page_url": "u",
                    "page_id": "p1",
                    "chunk_id": "c1",
                    "chunk_text": "context",
                    "score": 1.0,
                }
            ],
        )
        monkeypatch.setattr(retrieval_reranker, "rerank", lambda q, rows, text_key, top_k: rows)
        monkeypatch.setattr(fusion, "_synthesize_answer", lambda q, snippets: "A")
        monkeypatch.setattr(fusion.settings, "multi_hop_expansion", False)

        result = fusion.query_graph("question", top_k=1)

        assert result.retrieval_tier == "generated"
        assert result.trace is not None
        assert result.trace["tier"] == "generated"
        assert result.trace["steps"][0]["kind"] == "retrieval"

    def test_query_graph_sets_wrrf_trace_on_fallback(self, monkeypatch) -> None:
        monkeypatch.setattr(fusion.settings, "model_mode", "api")
        monkeypatch.setattr(
            fusion,
            "_run_generated_query",
            lambda question, top_k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(
            fusion,
            "_run_fallback_query",
            lambda question, top_k: [
                {
                    "page_title": "P",
                    "page_url": "u",
                    "page_id": "p1",
                    "chunk_id": "c1",
                    "chunk_text": "fallback",
                    "score": 0.6,
                }
            ],
        )
        monkeypatch.setattr(retrieval_reranker, "rerank", lambda q, rows, text_key, top_k: rows)
        monkeypatch.setattr(fusion, "_synthesize_answer", lambda q, snippets: "A")
        monkeypatch.setattr(fusion.settings, "multi_hop_expansion", False)

        result = fusion.query_graph("question", top_k=1)

        assert result.retrieval_tier == "wrrf"
        assert result.trace is not None
        assert result.trace["tier"] == "wrrf"
        assert result.trace["steps"][0]["name"] == "generated_query_failed"


class TestMCPAuthMiddleware:
    def test_mcp_endpoint_blocked_without_auth(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", "secret123")

        with TestClient(main.app) as client:
            resp = client.get("/mcp")

        assert resp.status_code == 401

    def test_mcp_endpoint_allowed_with_correct_auth(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", "secret123")

        with TestClient(main.app, raise_server_exceptions=False) as client:
            resp = client.get("/mcp", headers={"Authorization": "Bearer secret123"})

        # Not 401 — auth passed, even if MCP handler errors due to missing lifespan
        assert resp.status_code != 401

    def test_mcp_endpoint_no_auth_required_when_no_key(self, monkeypatch) -> None:
        monkeypatch.setattr(main.settings, "app_api_key", None)

        with TestClient(main.app, raise_server_exceptions=False) as client:
            resp = client.get("/mcp")

        assert resp.status_code != 401


class TestReadyEndpointSuccess:
    @patch("src.main.neo4j_client")
    def test_ready_when_neo4j_ok(self, mock_neo4j, monkeypatch) -> None:
        mock_neo4j.verify_connectivity.return_value = None

        with TestClient(main.app) as client:
            resp = client.get("/ready")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["neo4j"]["ok"] is True
