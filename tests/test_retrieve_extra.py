"""Tests for retrieve module — fallback query, generated query, shape validation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

import src.retrieve as retrieve


class _FakeSession:
    def __init__(self, records=None, error=None):
        self._records = records or []
        self._error = error
        self.queries_run: list[str] = []

    def run(self, cypher: str, **params):
        self.queries_run.append(cypher)
        if self._error:
            raise self._error
        return self._records


class _FakeNeo4jClient:
    def __init__(self, records=None, error=None):
        self._records = records or []
        self._error = error
        self._session = _FakeSession(records, error)

    @contextmanager
    def session(self) -> Iterator[_FakeSession]:
        yield self._session


def _make_row(title="T", url="https://t", chunk_id="c1", text="chunk text", score=0.9):
    return {
        "page_title": title,
        "page_url": url,
        "chunk_id": chunk_id,
        "chunk_text": text,
        "score": score,
    }


class TestRunFallbackQuery:
    def test_returns_fused_rows(self, monkeypatch) -> None:
        rows = [_make_row()]
        monkeypatch.setattr(retrieve, "_run_bm25_query", lambda q, k: rows)
        monkeypatch.setattr(retrieve, "_run_vector_query", lambda q, k: [])
        monkeypatch.setattr(retrieve, "_run_graph_query", lambda q, k: [])
        monkeypatch.setattr(retrieve, "_community_search", lambda q, k: [])

        result = retrieve._run_fallback_query("test question", top_k=5)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c1"

    def test_falls_back_to_legacy_when_all_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(retrieve, "_run_bm25_query", lambda q, k: [])
        monkeypatch.setattr(retrieve, "_run_vector_query", lambda q, k: [])
        monkeypatch.setattr(retrieve, "_run_graph_query", lambda q, k: [])
        monkeypatch.setattr(retrieve, "_community_search", lambda q, k: [])

        legacy_rows = [_make_row(title="Legacy")]
        monkeypatch.setattr(retrieve, "neo4j_client", _FakeNeo4jClient(legacy_rows))

        result = retrieve._run_fallback_query("nothing", top_k=3)
        assert result == legacy_rows


class TestRunGeneratedQuery:
    def test_success_with_valid_cypher(self, monkeypatch) -> None:
        rows = [_make_row()]
        monkeypatch.setattr(retrieve, "generate_readonly_cypher", lambda _q: "MATCH (n) RETURN n")
        monkeypatch.setattr(retrieve, "assert_readonly_cypher", lambda _c: None)
        monkeypatch.setattr(retrieve, "neo4j_client", _FakeNeo4jClient(rows))

        result = retrieve._run_generated_query("test", top_k=4)
        assert result == rows

    def test_raises_on_invalid_shape(self, monkeypatch) -> None:
        bad_rows = [{"page_title": "T", "page_url": "U"}]  # missing keys
        monkeypatch.setattr(retrieve, "generate_readonly_cypher", lambda _q: "MATCH (n) RETURN n")
        monkeypatch.setattr(retrieve, "assert_readonly_cypher", lambda _c: None)
        monkeypatch.setattr(retrieve, "neo4j_client", _FakeNeo4jClient(bad_rows))

        with pytest.raises(RuntimeError, match="unexpected shape"):
            retrieve._run_generated_query("test", top_k=4)

    def test_raises_when_cypher_validation_fails(self, monkeypatch) -> None:
        monkeypatch.setattr(retrieve, "generate_readonly_cypher", lambda _q: "DELETE n")

        def _raise(cypher):
            raise RuntimeError("not read-only")

        monkeypatch.setattr(retrieve, "assert_readonly_cypher", _raise)

        with pytest.raises(RuntimeError, match="not read-only"):
            retrieve._run_generated_query("test", top_k=4)


class TestQueryGraph:
    def test_multiple_citations_and_answer_format(self, monkeypatch) -> None:
        rows = [
            _make_row(title="A", chunk_id="c1", text="First chunk"),
            _make_row(title="B", chunk_id="c2", text="Second chunk"),
        ]
        monkeypatch.setattr(retrieve, "generate_readonly_cypher", lambda _q: "MATCH")
        monkeypatch.setattr(retrieve, "assert_readonly_cypher", lambda _c: None)
        monkeypatch.setattr(retrieve, "neo4j_client", _FakeNeo4jClient(rows))

        result = retrieve.query_graph("test", top_k=5)

        assert len(result.citations) == 2
        assert result.citations[0]["chunk_id"] == "c1"
        assert "First chunk" in result.answer
        assert "Second chunk" in result.answer

    def test_fallback_on_generation_error(self, monkeypatch) -> None:
        fallback_rows = [_make_row(title="Fallback")]

        def _fail_generate(q):
            raise RuntimeError("Gemini down")

        monkeypatch.setattr(retrieve, "generate_readonly_cypher", _fail_generate)
        # WRRF fallback path: mock individual signal queries
        monkeypatch.setattr(retrieve, "_run_bm25_query", lambda q, k: fallback_rows)
        monkeypatch.setattr(retrieve, "_run_vector_query", lambda q, k: [])
        monkeypatch.setattr(retrieve, "_run_graph_query", lambda q, k: [])
        monkeypatch.setattr(retrieve, "_community_search", lambda q, k: [])

        result = retrieve.query_graph("test", top_k=3)
        assert result.citations[0]["page_title"] == "Fallback"

    def test_truncates_long_chunk_text_in_answer(self, monkeypatch) -> None:
        long_text = "x" * 500
        rows = [_make_row(text=long_text)]
        monkeypatch.setattr(retrieve, "generate_readonly_cypher", lambda _q: "MATCH")
        monkeypatch.setattr(retrieve, "assert_readonly_cypher", lambda _c: None)
        monkeypatch.setattr(retrieve, "neo4j_client", _FakeNeo4jClient(rows))

        result = retrieve.query_graph("test", top_k=1)
        assert len(result.answer) < 500
