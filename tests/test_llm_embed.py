"""Tests for embedding and Cypher generation in llm module."""

from __future__ import annotations

import pytest

import src.llm as llm


class _FakeEmbeddingResponse:
    def __init__(self, values):
        self.embeddings = [type("E", (), {"values": values})()]


class _FakeClient:
    def __init__(self, responses=None, error=None):
        self._responses = responses or []
        self._error = error
        self._call_count = 0

    @property
    def models(self):
        return self

    def embed_content(self, **kwargs):
        self._call_count += 1
        if self._error:
            raise self._error
        if self._responses:
            return self._responses.pop(0)
        return _FakeEmbeddingResponse([0.1, 0.2, 0.3])

    def generate_content(self, **kwargs):
        self._call_count += 1
        if self._error:
            raise self._error
        return type("R", (), {"text": '{"cypher":"MATCH (p:Page) RETURN p.title AS page_title, p.url AS page_url, p.id AS chunk_id, p.summary AS chunk_text, 1.0 AS score LIMIT $top_k"}'})()


class TestEmbedTextsLocal:
    def test_local_backend_uses_sentence_transformers(self, monkeypatch) -> None:
        monkeypatch.setattr(llm.settings, "embedding_backend", "local")

        class _FakeModel:
            def encode(self, texts, **kwargs):
                import numpy as np
                return [np.array([0.1, 0.2]) for _ in texts]

        monkeypatch.setattr(llm, "_local_embedding_model", _FakeModel())

        result = llm.embed_texts(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]


class TestEmbedTextsGemini:
    def test_success_on_first_key(self, monkeypatch) -> None:
        monkeypatch.setattr(llm.settings, "embedding_backend", "gemini")
        client = _FakeClient()
        monkeypatch.setattr(llm, "_client_pool", lambda: [client])

        result = llm.embed_texts(["test"])
        assert result == [[0.1, 0.2, 0.3]]

    def test_key_rotation_on_retryable_error(self, monkeypatch) -> None:
        monkeypatch.setattr(llm.settings, "embedding_backend", "gemini")
        monkeypatch.setattr(llm.time, "sleep", lambda _: None)

        bad_client = _FakeClient(error=RuntimeError("429 rate limit"))
        good_client = _FakeClient()
        monkeypatch.setattr(llm, "_client_pool", lambda: [bad_client, good_client])

        result = llm.embed_texts(["test"])
        assert result == [[0.1, 0.2, 0.3]]
        assert bad_client._call_count == 1
        assert good_client._call_count == 1

    def test_raises_on_non_retryable_error(self, monkeypatch) -> None:
        monkeypatch.setattr(llm.settings, "embedding_backend", "gemini")
        client = _FakeClient(error=ValueError("invalid input"))
        monkeypatch.setattr(llm, "_client_pool", lambda: [client])

        with pytest.raises(ValueError, match="invalid input"):
            llm.embed_texts(["test"])

    def test_raises_when_all_keys_exhausted(self, monkeypatch) -> None:
        monkeypatch.setattr(llm.settings, "embedding_backend", "gemini")
        monkeypatch.setattr(llm.time, "sleep", lambda _: None)

        bad1 = _FakeClient(error=RuntimeError("429 quota"))
        bad2 = _FakeClient(error=RuntimeError("rate limit"))
        monkeypatch.setattr(llm, "_client_pool", lambda: [bad1, bad2])

        with pytest.raises(RuntimeError, match="All Gemini keys failed"):
            llm.embed_texts(["test"])

    def test_raises_on_empty_embedding_response(self, monkeypatch) -> None:
        monkeypatch.setattr(llm.settings, "embedding_backend", "gemini")

        class _EmptyResp:
            embeddings = None

        client = _FakeClient(responses=[_EmptyResp()])
        monkeypatch.setattr(llm, "_client_pool", lambda: [client])

        with pytest.raises(RuntimeError, match="no vectors"):
            llm.embed_texts(["test"])


class TestGenerateReadonlyCypher:
    def test_success(self, monkeypatch) -> None:
        monkeypatch.setattr(llm.settings, "embedding_backend", "gemini")
        client = _FakeClient()
        monkeypatch.setattr(llm, "_client_pool", lambda: [client])

        cypher = llm.generate_readonly_cypher("What is Neo4j?")
        assert "page_title" in cypher
        assert "LIMIT" in cypher

    def test_key_rotation_on_rate_limit(self, monkeypatch) -> None:
        monkeypatch.setattr(llm.time, "sleep", lambda _: None)

        bad_client = _FakeClient(error=RuntimeError("429 rate limit"))
        good_client = _FakeClient()
        monkeypatch.setattr(llm, "_client_pool", lambda: [bad_client, good_client])

        cypher = llm.generate_readonly_cypher("test question")
        assert "page_title" in cypher

    def test_raises_when_all_keys_fail(self, monkeypatch) -> None:
        monkeypatch.setattr(llm.time, "sleep", lambda _: None)

        bad = _FakeClient(error=RuntimeError("quota exceeded"))
        monkeypatch.setattr(llm, "_client_pool", lambda: [bad])

        with pytest.raises(RuntimeError, match="All Gemini keys failed"):
            llm.generate_readonly_cypher("test")

    def test_adds_limit_when_missing(self, monkeypatch) -> None:
        class _NoLimitClient:
            @property
            def models(self):
                return self

            def generate_content(self, **kwargs):
                return type("R", (), {"text": '{"cypher":"MATCH (p:Page) RETURN p.title AS page_title, p.url AS page_url, p.id AS chunk_id, p.summary AS chunk_text, 1.0 AS score"}'})()

        monkeypatch.setattr(llm, "_client_pool", lambda: [_NoLimitClient()])

        cypher = llm.generate_readonly_cypher("test")
        assert "LIMIT $top_k" in cypher

    def test_raises_on_empty_cypher(self, monkeypatch) -> None:
        class _EmptyClient:
            @property
            def models(self):
                return self

            def generate_content(self, **kwargs):
                return type("R", (), {"text": '{"cypher":""}'})()

        monkeypatch.setattr(llm, "_client_pool", lambda: [_EmptyClient()])

        with pytest.raises(RuntimeError, match="empty Cypher"):
            llm.generate_readonly_cypher("test")
