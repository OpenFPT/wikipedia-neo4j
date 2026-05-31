"""Tests for the GraphPulse dashboard modules."""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.dashboard.query_log import QueryLog, QueryLogEntry, query_log


class TestQueryLog:
    def test_append_and_recent(self):
        log = QueryLog(maxlen=5)
        entry = QueryLogEntry(
            timestamp="2024-01-01T00:00:00",
            question="test?",
            retrieval_tier="hybrid",
            latency_ms=100,
            result_count=3,
            signal_scores={"bm25": 5, "vector": 3},
        )
        log.append(entry)
        recent = log.recent(10)
        assert len(recent) == 1
        assert recent[0]["question"] == "test?"
        assert recent[0]["signal_scores"] == {"bm25": 5, "vector": 3}

    def test_recent_returns_newest_first(self):
        log = QueryLog(maxlen=10)
        for i in range(5):
            log.append(
                QueryLogEntry(
                    timestamp=f"2024-01-0{i+1}",
                    question=f"q{i}",
                    retrieval_tier="hybrid",
                    latency_ms=i * 10,
                    result_count=i,
                )
            )
        recent = log.recent(3)
        assert len(recent) == 3
        assert recent[0]["question"] == "q4"

    def test_latest_empty(self):
        log = QueryLog()
        assert log.latest() is None

    def test_latest_returns_most_recent(self):
        log = QueryLog()
        log.append(
            QueryLogEntry(
                timestamp="t1",
                question="first",
                retrieval_tier="bm25",
                latency_ms=50,
                result_count=1,
            )
        )
        log.append(
            QueryLogEntry(
                timestamp="t2",
                question="second",
                retrieval_tier="vector",
                latency_ms=60,
                result_count=2,
            )
        )
        assert log.latest().question == "second"

    def test_len(self):
        log = QueryLog(maxlen=3)
        assert len(log) == 0
        for i in range(5):
            log.append(
                QueryLogEntry(
                    timestamp=f"t{i}",
                    question=f"q{i}",
                    retrieval_tier="hybrid",
                    latency_ms=10,
                    result_count=1,
                )
            )
        assert len(log) == 3

    def test_thread_safety(self):
        log = QueryLog(maxlen=100)
        errors = []

        def writer(start):
            try:
                for i in range(20):
                    log.append(
                        QueryLogEntry(
                            timestamp=f"t{start+i}",
                            question=f"q{start+i}",
                            retrieval_tier="hybrid",
                            latency_ms=10,
                            result_count=1,
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 20,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(log) == 100


class TestDashboardData:
    def test_fetch_graph_stats_failure(self):
        mock_client = MagicMock()
        mock_client.session.side_effect = Exception("connection refused")

        with patch(
            "src.infrastructure.neo4j_client.neo4j_client", mock_client
        ):
            from src.dashboard.data import fetch_graph_stats

            result = fetch_graph_stats()
            assert result["available"] is False
            assert result["pages"] is None

    def test_fetch_graph_stats_success(self):
        mock_record = {
            "pages": 100,
            "chunks": 500,
            "entities": 200,
            "persons": 50,
            "orgs": 30,
            "locations": 80,
            "works": 40,
            "has_chunk_rels": 500,
            "mention_rels": 300,
            "links_to_rels": 150,
        }
        mock_result = MagicMock()
        mock_result.single.return_value = mock_record
        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.session.return_value = mock_session

        with patch(
            "src.infrastructure.neo4j_client.neo4j_client", mock_client
        ):
            from src.dashboard.data import fetch_graph_stats

            result = fetch_graph_stats()
            assert result["available"] is True
            assert result["pages"] == 100
            assert result["total_rels"] == 950

    def test_fetch_recent_queries(self):
        from src.dashboard.data import fetch_recent_queries

        query_log._buffer.clear()
        query_log.append(
            QueryLogEntry(
                timestamp="t1",
                question="hello",
                retrieval_tier="hybrid",
                latency_ms=50,
                result_count=3,
            )
        )
        result = fetch_recent_queries(5)
        assert len(result) == 1
        assert result[0]["question"] == "hello"
        query_log._buffer.clear()

    def test_fetch_signal_breakdown_none(self):
        from src.dashboard.data import fetch_signal_breakdown

        query_log._buffer.clear()
        assert fetch_signal_breakdown() is None

    def test_fetch_signal_breakdown_with_data(self):
        from src.dashboard.data import fetch_signal_breakdown

        query_log._buffer.clear()
        query_log.append(
            QueryLogEntry(
                timestamp="t1",
                question="test q",
                retrieval_tier="hybrid",
                latency_ms=50,
                result_count=3,
                signal_scores={"bm25": 10, "vector": 8},
            )
        )
        result = fetch_signal_breakdown()
        assert result is not None
        assert result["question"] == "test q"
        assert result["scores"] == {"bm25": 10, "vector": 8}
        query_log._buffer.clear()

    def test_fetch_signal_breakdown_empty_scores(self):
        from src.dashboard.data import fetch_signal_breakdown

        query_log._buffer.clear()
        query_log.append(
            QueryLogEntry(
                timestamp="t1",
                question="no signals",
                retrieval_tier="hybrid",
                latency_ms=50,
                result_count=3,
                signal_scores={},
            )
        )
        result = fetch_signal_breakdown()
        assert result is None
        query_log._buffer.clear()

    def test_fetch_eval_metrics_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from src.dashboard.data import fetch_eval_metrics

        assert fetch_eval_metrics() is None

    def test_fetch_eval_metrics_valid(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        eval_data = {
            "timestamp": "2024-01-01",
            "total": 100,
            "context_hit_rate": 0.85,
            "mrr": 0.72,
            "rerank_context_hit_rate": 0.90,
            "rerank_mrr": 0.78,
            "avg_latency_ms": 150,
        }
        (data_dir / "eval_results.json").write_text(json.dumps(eval_data))

        from src.dashboard.data import fetch_eval_metrics

        result = fetch_eval_metrics()
        assert result is not None
        assert result["available"] is True
        assert result["context_hit_rate"] == 0.85

    def test_fetch_eval_metrics_invalid_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "eval_results.json").write_text("not json{{{")

        from src.dashboard.data import fetch_eval_metrics

        assert fetch_eval_metrics() is None

    def test_fetch_wrrf_weights(self):
        from src.dashboard.data import fetch_wrrf_weights

        weights = fetch_wrrf_weights()
        assert "bm25" in weights
        assert "vector" in weights
        assert "graph" in weights
        assert "community" in weights


class TestDashboardRoutes:
    @pytest.fixture
    def client(self):
        from src.main import app

        return TestClient(app)

    def test_dashboard_page(self, client):
        with patch("src.dashboard.routes.fetch_graph_stats") as mock_stats:
            mock_stats.return_value = {
                "pages": 10,
                "chunks": 50,
                "entities": 20,
                "persons": 5,
                "orgs": 3,
                "locations": 8,
                "works": 4,
                "has_chunk_rels": 50,
                "mention_rels": 30,
                "links_to_rels": 10,
                "total_rels": 90,
                "available": True,
            }
            resp = client.get("/dashboard")
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]

    def test_api_stats(self, client):
        with patch("src.dashboard.routes.fetch_graph_stats") as mock_stats:
            mock_stats.return_value = {"pages": 5, "available": True}
            resp = client.get("/dashboard/api/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert data["pages"] == 5

    def test_api_queries(self, client):
        with patch("src.dashboard.routes.fetch_recent_queries") as mock_q:
            mock_q.return_value = [{"question": "test"}]
            resp = client.get("/dashboard/api/queries")
            assert resp.status_code == 200
            assert resp.json()["queries"] == [{"question": "test"}]

    def test_api_signals_empty(self, client):
        with patch("src.dashboard.routes.fetch_signal_breakdown") as mock_s:
            mock_s.return_value = None
            resp = client.get("/dashboard/api/signals")
            assert resp.status_code == 200
            assert resp.json() == {"scores": None}

    def test_api_signals_with_data(self, client):
        with patch("src.dashboard.routes.fetch_signal_breakdown") as mock_s:
            mock_s.return_value = {
                "scores": {"bm25": 5},
                "question": "q",
                "timestamp": "t",
            }
            resp = client.get("/dashboard/api/signals")
            assert resp.status_code == 200
            assert resp.json()["scores"] == {"bm25": 5}

    def test_api_eval_empty(self, client):
        with patch("src.dashboard.routes.fetch_eval_metrics") as mock_e:
            mock_e.return_value = None
            resp = client.get("/dashboard/api/eval")
            assert resp.status_code == 200
            assert resp.json() == {"available": False}

    def test_api_eval_with_data(self, client):
        with patch("src.dashboard.routes.fetch_eval_metrics") as mock_e:
            mock_e.return_value = {"available": True, "mrr": 0.75}
            resp = client.get("/dashboard/api/eval")
            assert resp.status_code == 200
            assert resp.json()["mrr"] == 0.75
