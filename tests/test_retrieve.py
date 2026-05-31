from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import src.retrieval.hybrid as retrieve


class _FakeSession:
    def __init__(self, records):
        self._records = records

    def run(self, _cypher: str, **_params):
        return self._records


class _FakeNeo4jClient:
    def __init__(self, records):
        self._records = records

    @contextmanager
    def session(self) -> Iterator[_FakeSession]:
        yield _FakeSession(self._records)


def test_query_graph_uses_generated_query_when_valid(monkeypatch) -> None:
    rows = [
        {
            "page_title": "Neo4j",
            "page_url": "https://en.wikipedia.org/wiki/Neo4j",
            "chunk_id": "c1",
            "chunk_text": "Neo4j is a graph database system.",
            "score": 0.91,
        }
    ]

    monkeypatch.setattr(retrieve, "generate_readonly_cypher", lambda _q: "MATCH (n) RETURN n")
    monkeypatch.setattr(retrieve, "assert_readonly_cypher", lambda _c: None)
    monkeypatch.setattr(retrieve, "neo4j_client", _FakeNeo4jClient(rows))

    result = retrieve.query_graph("What is Neo4j?", top_k=3)

    assert result.citations == [
        {
            "page_title": "Neo4j",
            "page_url": "https://en.wikipedia.org/wiki/Neo4j",
            "chunk_id": "c1",
        }
    ]
    assert result.answer.startswith("Dựa trên thông tin tìm được:")
    assert result.retrieval_tier == "generated"


def test_query_graph_falls_back_when_generated_query_invalid(monkeypatch) -> None:
    fallback_rows = [
        {
            "page_title": "Fallback",
            "page_url": "https://example.org/fallback",
            "chunk_id": "f1",
            "chunk_text": "Fallback text",
            "score": 0.5,
        }
    ]

    monkeypatch.setattr(retrieve, "generate_readonly_cypher", lambda _q: "bad")

    def _raise(_cypher: str) -> None:
        raise RuntimeError("bad cypher")

    monkeypatch.setattr(retrieve, "assert_readonly_cypher", _raise)
    monkeypatch.setattr(retrieve, "neo4j_client", _FakeNeo4jClient(fallback_rows))

    result = retrieve.query_graph("Any", top_k=2)

    assert result.citations[0]["chunk_id"] == "f1"


def test_query_graph_returns_empty_when_no_rows(monkeypatch) -> None:
    monkeypatch.setattr(retrieve, "generate_readonly_cypher", lambda _q: "MATCH")
    monkeypatch.setattr(retrieve, "assert_readonly_cypher", lambda _c: None)
    monkeypatch.setattr(retrieve, "neo4j_client", _FakeNeo4jClient([]))

    result = retrieve.query_graph("Any", top_k=2)

    assert result.citations == []
    assert "could not find relevant context" in result.answer.lower()
