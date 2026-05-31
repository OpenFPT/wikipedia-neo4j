from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import src.retrieval.hybrid as retrieve


class _FakeSession:
    def __init__(self):
        self.cypher = ""

    def run(self, cypher: str, **_params):
        self.cypher = cypher
        return [
            {
                "page_title": "Neo4j",
                "page_url": "https://en.wikipedia.org/wiki/Neo4j",
                "chunk_id": "chunk-1",
                "chunk_text": "Neo4j is a native graph database.",
                "score": 1.0,
            }
        ]


class _FakeNeo4jClient:
    def __init__(self) -> None:
        self.last_session: _FakeSession | None = None

    @contextmanager
    def session(self) -> Iterator[_FakeSession]:
        sess = _FakeSession()
        self.last_session = sess
        yield sess


def test_hybrid_fallback_query_shape(monkeypatch) -> None:
    fake_client = _FakeNeo4jClient()

    def _raise(_q: str) -> str:
        raise RuntimeError("force fallback")

    monkeypatch.setattr(retrieve, "generate_readonly_cypher", _raise)
    monkeypatch.setattr(retrieve, "neo4j_client", fake_client)

    result = retrieve.query_graph("What is Neo4j?", top_k=3)

    # Verify we got a result with proper structure
    assert result.citations is not None
    assert len(result.citations) > 0
    assert result.citations[0]["chunk_id"] == "chunk-1"
    assert result.answer is not None
