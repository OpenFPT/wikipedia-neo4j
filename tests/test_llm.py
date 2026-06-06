import json

import pytest

from src.infrastructure.llm import assert_readonly_cypher


def test_assert_readonly_cypher_accepts_valid_aliases() -> None:
    cypher = """
    MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk)
    RETURN p.title AS page_title,
           p.url AS page_url,
           c.id AS chunk_id,
           c.text AS chunk_text,
           1.0 AS score
    LIMIT $top_k
    """
    assert_readonly_cypher(cypher)


@pytest.mark.parametrize(
    "cypher, expected_msg",
    [
        ("", "empty"),
        (
            "MATCH (p:Page) RETURN p.title AS page_title; MATCH (n) RETURN n",
            "multiple statements",
        ),
        (
            "MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk) "
            "RETURN p.title AS page_title, p.url AS page_url, c.id AS chunk_id, 1.0 AS score",
            "missing required alias",
        ),
        (
            "MATCH (p:Page) DELETE p RETURN '' AS page_title, '' AS page_url, '' AS chunk_id, '' AS chunk_text, 0 AS score",
            "not read-only",
        ),
    ],
)
def test_assert_readonly_cypher_rejects_invalid_input(cypher: str, expected_msg: str) -> None:
    with pytest.raises(RuntimeError, match=expected_msg):
        assert_readonly_cypher(cypher)


class TestEmbedTextsBatch:
    def test_batches_correctly(self, monkeypatch):
        from src.infrastructure import llm as llm_mod

        call_count = [0]

        def _fake_embed(texts):
            call_count[0] += 1
            return [[0.1] * 10 for _ in texts]

        monkeypatch.setattr(llm_mod, "embed_texts", _fake_embed)
        monkeypatch.setattr(llm_mod.settings, "embed_batch_size", 2)

        result = llm_mod.embed_texts_batch(["a", "b", "c", "d", "e"], batch_size=2, pause_between_batches=0)
        assert len(result) == 5
        assert call_count[0] == 3

    def test_uses_default_batch_size(self, monkeypatch):
        from src.infrastructure import llm as llm_mod

        def _fake_embed(texts):
            return [[0.1] * 10 for _ in texts]

        monkeypatch.setattr(llm_mod, "embed_texts", _fake_embed)
        monkeypatch.setattr(llm_mod.settings, "embed_batch_size", 50)

        result = llm_mod.embed_texts_batch(["a", "b", "c"])
        assert len(result) == 3

    def test_no_pause_for_local_backend(self, monkeypatch):
        from src.infrastructure import llm as llm_mod
        import time

        pauses = []

        def _track_sleep(s):
            pauses.append(s)

        def _fake_embed(texts):
            return [[0.1] * 10 for _ in texts]

        monkeypatch.setattr(llm_mod, "embed_texts", _fake_embed)
        monkeypatch.setattr(llm_mod.settings, "embedding_backend", "local")
        monkeypatch.setattr(time, "sleep", _track_sleep)

        result = llm_mod.embed_texts_batch(["a", "b", "c"], batch_size=1, pause_between_batches=1.0)
        assert len(result) == 3
        assert len(pauses) == 0


class TestGenerateCypherLocal:
    def test_generates_valid_cypher(self, monkeypatch):
        from src.infrastructure import llm as llm_mod

        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            return json.dumps({"cypher": "MATCH (p:Page) RETURN p.title AS page_title LIMIT $top_k"})

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = llm_mod._generate_cypher_local("test question")
        assert "MATCH" in result
        assert "$top_k" in result

    def test_appends_limit_if_missing(self, monkeypatch):
        from src.infrastructure import llm as llm_mod

        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            return json.dumps({"cypher": "MATCH (p:Page) RETURN p.title AS page_title"})

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = llm_mod._generate_cypher_local("test")
        assert "LIMIT $top_k" in result

    def test_raises_on_empty_cypher(self, monkeypatch):
        from src.infrastructure import llm as llm_mod

        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            return json.dumps({"cypher": ""})

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        with pytest.raises(RuntimeError, match="empty Cypher"):
            llm_mod._generate_cypher_local("test")

    def test_handles_code_fence(self, monkeypatch):
        from src.infrastructure import llm as llm_mod

        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            return '```json\n{"cypher": "MATCH (n) RETURN n LIMIT $top_k"}\n```'

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = llm_mod._generate_cypher_local("test")
        assert "MATCH" in result


class TestGenerateReadonlyCypher:
    def test_uses_local_mode(self, monkeypatch):
        from src.infrastructure import llm as llm_mod

        monkeypatch.setattr(llm_mod.settings, "model_mode", "local")

        def _fake_local(question):
            return "MATCH (n) RETURN n LIMIT $top_k"

        monkeypatch.setattr(llm_mod, "_generate_cypher_local", _fake_local)

        result = llm_mod.generate_readonly_cypher("test")
        assert "MATCH" in result
