"""Tests for cross-encoder reranker module."""

from __future__ import annotations

import numpy as np
import pytest

import src.reranker as reranker_mod


_real_rerank = reranker_mod.rerank


@pytest.fixture(autouse=True)
def _reset_reranker(monkeypatch):
    """Override the global _mock_reranker fixture and restore real rerank."""
    monkeypatch.setattr(reranker_mod, "_reranker", None)
    monkeypatch.setattr(reranker_mod, "rerank", _real_rerank)


class _FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return np.array([1.0 / (i + 1) for i in range(len(pairs))])


class TestRerank:
    def test_empty_documents(self, monkeypatch) -> None:
        result = reranker_mod.rerank("query", [], top_k=3)
        assert result == []

    def test_reranks_and_truncates(self, monkeypatch) -> None:
        monkeypatch.setattr(reranker_mod, "_get_reranker", lambda: _FakeCrossEncoder())

        docs = [
            {"chunk_text": "third best", "id": 3},
            {"chunk_text": "best match", "id": 1},
            {"chunk_text": "second best", "id": 2},
        ]

        result = reranker_mod.rerank("query", docs, text_key="chunk_text", top_k=2)

        assert len(result) == 2
        assert all("rerank_score" in d for d in result)
        assert result[0]["rerank_score"] >= result[1]["rerank_score"]

    def test_uses_text_key(self, monkeypatch) -> None:
        monkeypatch.setattr(reranker_mod, "_get_reranker", lambda: _FakeCrossEncoder())

        docs = [{"content": "hello", "id": 1}]
        result = reranker_mod.rerank("q", docs, text_key="content", top_k=5)
        assert len(result) == 1

    def test_get_reranker_loads_model(self, monkeypatch) -> None:
        monkeypatch.setattr(reranker_mod, "CrossEncoder", _FakeCrossEncoder)
        monkeypatch.setattr(reranker_mod, "_reranker", None)

        model = reranker_mod._get_reranker()
        assert model is not None
        assert reranker_mod._reranker is not None
