"""Tests for hybrid retrieval functions in src/retrieve.py."""

from __future__ import annotations

from contextlib import contextmanager


import src.retrieval.hybrid as retrieve_mod
from src.retrieval.hybrid import _wrrf_fuse, _vector_search, _graph_search, hybrid_retrieve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(chunk_id: str, page_title: str = "Page", score_key: str = "score", score: float = 1.0) -> dict:
    row = {
        "page_title": page_title,
        "page_url": f"https://vi.wikipedia.org/wiki/{page_title}",
        "page_id": f"pid_{chunk_id}",
        "chunk_id": chunk_id,
        "chunk_text": f"Text for {chunk_id}",
    }
    row[score_key] = score
    return row


class _FakeSession:
    def __init__(self, results: list[dict]):
        self._results = results

    def run(self, cypher, **params):
        return self._results

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _make_fake_session_factory(results: list[dict]):
    @contextmanager
    def _fake_session():
        yield _FakeSession(results)

    return _fake_session


# ---------------------------------------------------------------------------
# Tests: _wrrf_fuse (named-channel version)
# ---------------------------------------------------------------------------


class TestWrrfFuse:
    """Tests for the named-channel _wrrf_fuse function."""

    def test_basic_fusion_three_channels(self, monkeypatch) -> None:
        """Verify WRRF formula with 3 channels and known rankings."""
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_bm25", 0.4)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_vector", 0.4)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_graph", 0.2)

        bm25 = [_make_row("c1", "P1"), _make_row("c2", "P2")]
        vector = [_make_row("c2", "P2"), _make_row("c3", "P3")]
        graph = [_make_row("c1", "P1"), _make_row("c3", "P3")]

        results_by_channel = {"bm25": bm25, "vector": vector, "graph": graph}
        k = 60

        results = _wrrf_fuse(results_by_channel, k=k)

        scores = {r["chunk_id"]: r["score"] for r in results}

        # c1: bm25 rank=1, graph rank=1 -> 0.4/(60+1) + 0.2/(60+1)
        expected_c1 = 0.4 / (60 + 1) + 0.2 / (60 + 1)
        # c2: bm25 rank=2, vector rank=1 -> 0.4/(60+2) + 0.4/(60+1)
        expected_c2 = 0.4 / (60 + 2) + 0.4 / (60 + 1)
        # c3: vector rank=2, graph rank=2 -> 0.4/(60+2) + 0.2/(60+2)
        expected_c3 = 0.4 / (60 + 2) + 0.2 / (60 + 2)

        assert abs(scores["c1"] - expected_c1) < 1e-9
        assert abs(scores["c2"] - expected_c2) < 1e-9
        assert abs(scores["c3"] - expected_c3) < 1e-9

    def test_includes_channel_scores(self, monkeypatch) -> None:
        """Each result should have channel_scores showing per-channel contribution."""
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_bm25", 0.5)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_vector", 0.5)

        results_by_channel = {
            "bm25": [_make_row("c1")],
            "vector": [_make_row("c1")],
        }

        results = _wrrf_fuse(results_by_channel, k=60)

        assert len(results) == 1
        assert "channel_scores" in results[0]
        assert "bm25" in results[0]["channel_scores"]
        assert "vector" in results[0]["channel_scores"]

    def test_empty_channels_returns_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_bm25", 0.4)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_vector", 0.4)

        results = _wrrf_fuse({"bm25": [], "vector": []}, k=60)
        assert results == []

    def test_single_channel_nonempty(self, monkeypatch) -> None:
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_bm25", 0.4)

        bm25 = [_make_row("c1"), _make_row("c2"), _make_row("c3")]
        results = _wrrf_fuse({"bm25": bm25}, k=60)

        assert len(results) == 3
        # Order preserved from single channel
        assert results[0]["chunk_id"] == "c1"
        assert results[1]["chunk_id"] == "c2"
        assert results[2]["chunk_id"] == "c3"

    def test_deduplication(self, monkeypatch) -> None:
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_bm25", 0.4)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_vector", 0.4)

        bm25 = [_make_row("c1", "PageA"), _make_row("c2", "PageB")]
        vector = [_make_row("c1", "PageA_vec"), _make_row("c2", "PageB_vec")]

        results = _wrrf_fuse({"bm25": bm25, "vector": vector}, k=60)

        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_community_channel_weight(self, monkeypatch) -> None:
        """Community channel uses its own weight from settings."""
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_bm25", 0.0)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_community", 1.0)

        results_by_channel = {
            "bm25": [_make_row("c1")],
            "community": [_make_row("c2")],
        }

        results = _wrrf_fuse(results_by_channel, k=60)

        scores = {r["chunk_id"]: r["score"] for r in results}
        # c2 from community with weight 1.0 should score higher than c1 from bm25 with weight 0.0
        assert scores["c2"] > scores["c1"]

    def test_results_sorted_descending(self, monkeypatch) -> None:
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_bm25", 0.4)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_vector", 0.4)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_graph", 0.2)

        bm25 = [_make_row("c1"), _make_row("c2"), _make_row("c3")]
        vector = [_make_row("c3"), _make_row("c1"), _make_row("c2")]
        graph = [_make_row("c2"), _make_row("c3"), _make_row("c1")]

        results = _wrrf_fuse({"bm25": bm25, "vector": vector, "graph": graph}, k=60)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Tests: _vector_search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    def test_returns_results_with_legacy_cypher(self, monkeypatch) -> None:
        monkeypatch.setattr(retrieve_mod.settings, "neo4j_use_search_clause", False)

        fake_rows = [
            {"page_title": "P1", "page_url": "http://p1", "page_id": "pid1",
             "chunk_id": "c1", "chunk_text": "text1", "vector_score": 0.95},
        ]
        fake = _make_fake_session_factory(fake_rows)
        monkeypatch.setattr(retrieve_mod.neo4j_client, "session", fake)

        results = _vector_search([0.1] * 1024, top_k=5)

        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["vector_score"] == 0.95

    def test_returns_results_with_search_clause(self, monkeypatch) -> None:
        monkeypatch.setattr(retrieve_mod.settings, "neo4j_use_search_clause", True)

        fake_rows = [
            {"page_title": "P2", "page_url": "http://p2", "page_id": "pid2",
             "chunk_id": "c2", "chunk_text": "text2", "vector_score": 0.88},
        ]
        fake = _make_fake_session_factory(fake_rows)
        monkeypatch.setattr(retrieve_mod.neo4j_client, "session", fake)

        results = _vector_search([0.2] * 1024, top_k=5)

        assert len(results) == 1
        assert results[0]["chunk_id"] == "c2"

    def test_returns_empty_for_empty_embedding(self) -> None:
        results = _vector_search([], top_k=5)
        assert results == []

    def test_handles_neo4j_exception(self, monkeypatch) -> None:
        monkeypatch.setattr(retrieve_mod.settings, "neo4j_use_search_clause", False)

        @contextmanager
        def _failing_session():
            raise RuntimeError("Connection failed")
            yield  

        monkeypatch.setattr(retrieve_mod.neo4j_client, "session", _failing_session)

        results = _vector_search([0.1] * 10, top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: _graph_search
# ---------------------------------------------------------------------------


class TestGraphSearch:
    def test_returns_results(self, monkeypatch) -> None:
        fake_rows = [
            {"page_title": "Entity Page", "page_url": "http://ep", "page_id": "pid_ep",
             "chunk_id": "c_ent", "chunk_text": "Entity text", "graph_score": 3.5},
        ]
        fake = _make_fake_session_factory(fake_rows)
        monkeypatch.setattr(retrieve_mod.neo4j_client, "session", fake)

        results = _graph_search("Hà Nội", top_k=5)

        assert len(results) == 1
        assert results[0]["chunk_id"] == "c_ent"
        assert results[0]["graph_score"] == 3.5

    def test_returns_empty_on_no_match(self, monkeypatch) -> None:
        fake = _make_fake_session_factory([])
        monkeypatch.setattr(retrieve_mod.neo4j_client, "session", fake)

        results = _graph_search("nonexistent entity", top_k=5)
        assert results == []

    def test_handles_exception(self, monkeypatch) -> None:
        @contextmanager
        def _failing_session():
            raise RuntimeError("Neo4j down")
            yield  

        monkeypatch.setattr(retrieve_mod.neo4j_client, "session", _failing_session)

        results = _graph_search("test", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: hybrid_retrieve
# ---------------------------------------------------------------------------


class TestHybridRetrieve:
    def test_combines_all_channels(self, monkeypatch) -> None:
        """hybrid_retrieve should call BM25, vector, graph, and community channels."""
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_bm25", 0.4)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_vector", 0.4)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_graph", 0.2)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_community", 0.15)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_k", 60)

        # Mock embed_texts
        monkeypatch.setattr(retrieve_mod, "embed_texts", lambda texts: [[0.1] * 10])

        # Mock individual search functions
        bm25_results = [_make_row("c1", "BM25 Page", "bm25_score", 5.0)]
        vector_results = [_make_row("c2", "Vector Page", "vector_score", 0.9)]
        graph_results = [_make_row("c3", "Graph Page", "graph_score", 3.0)]
        community_results = [_make_row("c4", "Community Page", "score", 0.8)]

        monkeypatch.setattr(retrieve_mod, "_run_bm25_query", lambda q, k: bm25_results)
        monkeypatch.setattr(retrieve_mod, "_vector_search", lambda emb, k: vector_results)
        monkeypatch.setattr(retrieve_mod, "_graph_search", lambda q, k: graph_results)
        monkeypatch.setattr(retrieve_mod, "_community_search", lambda q, k, query_embedding=None: community_results)

        results = hybrid_retrieve("Thủ đô Việt Nam", top_k=10)

        # All 4 chunks should appear in fused results
        chunk_ids = {r["chunk_id"] for r in results}
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids
        assert "c3" in chunk_ids
        assert "c4" in chunk_ids

    def test_returns_empty_when_all_channels_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(retrieve_mod, "embed_texts", lambda texts: [[0.1] * 10])
        monkeypatch.setattr(retrieve_mod, "_run_bm25_query", lambda q, k: [])
        monkeypatch.setattr(retrieve_mod, "_vector_search", lambda emb, k: [])
        monkeypatch.setattr(retrieve_mod, "_graph_search", lambda q, k: [])
        monkeypatch.setattr(retrieve_mod, "_community_search", lambda q, k, query_embedding=None: [])

        results = hybrid_retrieve("nothing", top_k=10)
        assert results == []

    def test_works_without_embedding(self, monkeypatch) -> None:
        """If embedding fails, vector and community channels are skipped."""
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_bm25", 0.4)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_graph", 0.2)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_k", 60)

        def _fail_embed(texts):
            raise RuntimeError("Embedding service down")

        monkeypatch.setattr(retrieve_mod, "embed_texts", _fail_embed)

        bm25_results = [_make_row("c1", "BM25")]
        graph_results = [_make_row("c2", "Graph")]

        monkeypatch.setattr(retrieve_mod, "_run_bm25_query", lambda q, k: bm25_results)
        monkeypatch.setattr(retrieve_mod, "_graph_search", lambda q, k: graph_results)

        results = hybrid_retrieve("test query", top_k=10)

        chunk_ids = {r["chunk_id"] for r in results}
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids

    def test_respects_top_k(self, monkeypatch) -> None:
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_weight_bm25", 0.4)
        monkeypatch.setattr(retrieve_mod.settings, "wrrf_k", 60)

        monkeypatch.setattr(retrieve_mod, "embed_texts", lambda texts: [[0.1] * 10])

        bm25_results = [_make_row(f"c{i}", f"P{i}") for i in range(20)]
        monkeypatch.setattr(retrieve_mod, "_run_bm25_query", lambda q, k: bm25_results)
        monkeypatch.setattr(retrieve_mod, "_vector_search", lambda emb, k: [])
        monkeypatch.setattr(retrieve_mod, "_graph_search", lambda q, k: [])
        monkeypatch.setattr(retrieve_mod, "_community_search", lambda q, k, query_embedding=None: [])

        results = hybrid_retrieve("test", top_k=5)
        assert len(results) <= 5
