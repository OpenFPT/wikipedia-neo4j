"""Tests for entity extraction and classification in ingest module."""

from __future__ import annotations

import pytest

import src.ingest as ingest
import src.ner as ner


class TestClassifyEntityType:
    def test_organization_keywords(self) -> None:
        assert ner.classify_entity_type("Google Corporation") == "Organization"
        assert ner.classify_entity_type("Harvard University") == "Organization"
        assert ner.classify_entity_type("Tập đoàn Vingroup") == "Organization"
        assert ner.classify_entity_type("Đại học Bách Khoa") == "Organization"

    def test_location_keywords(self) -> None:
        assert ner.classify_entity_type("Ho Chi Minh City") == "Location"
        assert ner.classify_entity_type("Mekong River") == "Location"
        assert ner.classify_entity_type("Tỉnh Bình Dương") == "Location"
        assert ner.classify_entity_type("Mount Everest") == "Location"

    def test_work_keywords(self) -> None:
        assert ner.classify_entity_type("The Great Film") == "Work"
        assert ner.classify_entity_type("Tiểu thuyết Số Đỏ") == "Work"
        assert ner.classify_entity_type("Dark Side Album") == "Work"

    def test_person_by_word_count(self) -> None:
        assert ner.classify_entity_type("John Smith") == "Person"
        assert ner.classify_entity_type("Nguyen Van A") == "Person"

    def test_unknown_single_word(self) -> None:
        assert ner.classify_entity_type("X") == "Unknown"

    def test_unknown_long_name(self) -> None:
        assert ner.classify_entity_type("A B C D E") == "Unknown"


class TestExtractEntitiesNormalized:
    def test_simple_backend_returns_tuples(self, monkeypatch) -> None:
        monkeypatch.setattr(ner.settings, "ner_backend", "simple")
        result = ner.extract_entities("Google Cloud is in New York City area")
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        names = [name for name, _ in result]
        assert "Google Cloud" in names or "New York City" in names

    def test_underthesea_backend_returns_tuples(self, monkeypatch) -> None:
        monkeypatch.setattr(ner.settings, "ner_backend", "underthesea")

        def _fake_underthesea(text, max_entities=25):
            return [("Hanoi", "Location"), ("Vietnam", "Location")]

        monkeypatch.setattr(ner, "_extract_entities_underthesea", _fake_underthesea)
        result = ner.extract_entities("Hanoi is in Vietnam")
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_phonlp_backend_returns_tuples(self, monkeypatch) -> None:
        monkeypatch.setattr(ner.settings, "ner_backend", "phonlp")

        def _fake_phonlp(text, max_entities=25):
            return [("Hà Nội", "Location"), ("Việt Nam", "Location")]

        monkeypatch.setattr(ner, "_extract_entities_phonlp", _fake_phonlp)
        result = ner.extract_entities("Hà Nội là thủ đô Việt Nam")
        assert result == [("Hà Nội", "Location"), ("Việt Nam", "Location")]


class TestExtractEntitiesSimple:
    def test_deduplication(self) -> None:
        text = "New York is great. New York is big."
        entities = ner._extract_entities_simple(text)
        assert entities.count("New York") == 1

    def test_max_entities_limit(self) -> None:
        text = " ".join(f"Entity{i} Name{i}" for i in range(50))
        entities = ner._extract_entities_simple(text, max_entities=5)
        assert len(entities) <= 5

    def test_skips_short_candidates(self) -> None:
        text = "A B is here but So is John Smith"
        entities = ner._extract_entities_simple(text)
        for e in entities:
            assert len(e) >= 3


class TestUpsertPageFromText:
    def test_upsert_calls_neo4j_correctly(self, monkeypatch) -> None:
        queries_run: list[str] = []

        class _FakeSession:
            def run(self, cypher, **params):
                queries_run.append(cypher.strip()[:20])

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        class _FakeClient:
            def setup_schema(self):
                pass

            def session(self):
                from contextlib import contextmanager

                @contextmanager
                def _s():
                    yield _FakeSession()

                return _s()

        monkeypatch.setattr(ingest, "neo4j_client", _FakeClient())
        monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[0.1] * 3 for _ in texts])
        monkeypatch.setattr(ner.settings, "ner_backend", "simple")

        result = ingest._upsert_page_from_text(
            page_id="p1",
            title="Test Page",
            url="https://test.org",
            text="John Smith works at Google Corporation in New York City area.",
            summary="A test page.",
        )

        assert result.page_id == "p1"
        assert result.title == "Test Page"
        assert result.chunk_count >= 1
        assert result.entity_count >= 1
        assert len(queries_run) > 0


class TestIngestTopic:
    def test_disambiguation_error_raises_value_error(self, monkeypatch) -> None:
        def _raise(*_args, **_kwargs):
            raise ingest.wikipedia.exceptions.DisambiguationError("Python", ["Python (programming)", "Python (snake)"])

        monkeypatch.setattr(ingest.wikipedia, "page", _raise)

        with pytest.raises(ValueError, match="Ambiguous topic"):
            ingest.ingest_topic("Python")

    def test_page_error_raises_value_error(self, monkeypatch) -> None:
        def _raise(*_args, **_kwargs):
            raise ingest.wikipedia.exceptions.PageError(pageid="xyz")

        monkeypatch.setattr(ingest.wikipedia, "page", _raise)

        with pytest.raises(ValueError, match="page not found"):
            ingest.ingest_topic("NonexistentPage12345")

    def test_successful_ingestion(self, monkeypatch) -> None:
        class _FakePage:
            title = "Neo4j"
            url = "https://en.wikipedia.org/wiki/Neo4j"
            content = "Neo4j is a graph database management system."
            links = ["Graph database", "NoSQL"]

        monkeypatch.setattr(ingest.wikipedia, "page", lambda *a, **kw: _FakePage())
        monkeypatch.setattr(ingest.wikipedia, "summary", lambda *a, **kw: "Neo4j summary")
        monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[0.1] for _ in texts])
        monkeypatch.setattr(ner.settings, "ner_backend", "simple")

        queries: list[str] = []

        class _FakeSession:
            def run(self, cypher, **params):
                queries.append(cypher.strip()[:30])

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        class _FakeClient:
            def setup_schema(self):
                pass

            def session(self):
                from contextlib import contextmanager

                @contextmanager
                def _s():
                    yield _FakeSession()

                return _s()

        monkeypatch.setattr(ingest, "neo4j_client", _FakeClient())

        result = ingest.ingest_topic("Neo4j")

        assert result.title == "Neo4j"
        assert result.url == "https://en.wikipedia.org/wiki/Neo4j"
        assert result.chunk_count >= 1

    def test_summary_fallback_on_error(self, monkeypatch) -> None:
        class _FakePage:
            title = "Test"
            url = "https://en.wikipedia.org/wiki/Test"
            content = "A" * 500
            links = []

        monkeypatch.setattr(ingest.wikipedia, "page", lambda *a, **kw: _FakePage())
        monkeypatch.setattr(ingest.wikipedia, "summary", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail")))
        monkeypatch.setattr(ingest, "embed_texts", lambda texts: [[0.1] for _ in texts])
        monkeypatch.setattr(ner.settings, "ner_backend", "simple")

        queries: list[str] = []

        class _FakeSession:
            def run(self, cypher, **params):
                queries.append(cypher)

        class _FakeClient:
            def setup_schema(self):
                pass

            def session(self):
                from contextlib import contextmanager

                @contextmanager
                def _s():
                    yield _FakeSession()

                return _s()

        monkeypatch.setattr(ingest, "neo4j_client", _FakeClient())

        result = ingest.ingest_topic("Test")
        assert result.title == "Test"
