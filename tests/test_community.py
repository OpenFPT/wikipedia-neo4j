"""Tests for src/community.py — community-based retrieval."""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

import src.community as community_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRecord:
    """Mimics a Neo4j Record supporting both [] access and .single()."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)


class _FakeResult:
    """Mimics a Neo4j Result supporting iteration and .single()."""

    def __init__(self, records: list[dict]):
        self._records = records

    def single(self):
        if self._records:
            return _FakeRecord(self._records[0])
        return None

    def __iter__(self):
        return iter([_FakeRecord(r) for r in self._records])


class _FakeSession:
    """Fake session that returns pre-configured results for sequential run() calls."""

    def __init__(self, results_queue: list):
        self._results_queue = results_queue
        self._call_idx = 0

    def run(self, cypher, **params):
        if self._call_idx < len(self._results_queue):
            records = self._results_queue[self._call_idx]
            self._call_idx += 1
            return _FakeResult(records)
        self._call_idx += 1
        return _FakeResult([])

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _make_shared_session_factory(results_queue: list):
    """Create a session factory where ALL context manager entries share the same call queue."""
    session = _FakeSession(results_queue)

    @contextmanager
    def _fake_session():
        yield session

    return _fake_session


# ---------------------------------------------------------------------------
# Tests: get_community_for_entity
# ---------------------------------------------------------------------------


class TestGetCommunityForEntity:
    def test_returns_community_id(self, monkeypatch) -> None:
        fake = _make_shared_session_factory([[{"community_id": 42}]])
        monkeypatch.setattr(community_mod.neo4j_client, "session", fake)

        result = community_mod.get_community_for_entity("Hà Nội")
        assert result == 42

    def test_returns_none_when_not_found(self, monkeypatch) -> None:
        fake = _make_shared_session_factory([[]])
        monkeypatch.setattr(community_mod.neo4j_client, "session", fake)

        result = community_mod.get_community_for_entity("Nonexistent Entity")
        assert result is None

    def test_returns_none_when_community_id_is_null(self, monkeypatch) -> None:
        fake = _make_shared_session_factory([[{"community_id": None}]])
        monkeypatch.setattr(community_mod.neo4j_client, "session", fake)

        result = community_mod.get_community_for_entity("Entity Without Community")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: get_community_summary
# ---------------------------------------------------------------------------


class TestGetCommunitySummary:
    def test_returns_summary_from_jsonl_cache(self, monkeypatch, tmp_path) -> None:
        # Reset the singleton
        monkeypatch.setattr(community_mod, "_summaries", None)

        # Create a temp JSONL file
        summaries_file = tmp_path / "summaries.jsonl"
        record = {
            "community_id": "community_00005",
            "summary": "This community is about Vietnamese geography.",
            "member_count": 10,
            "top_entities": ["Hà Nội", "Sài Gòn"],
            "entity_ids": ["e1", "e2"],
        }
        summaries_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

        monkeypatch.setattr(community_mod, "_SUMMARIES_PATH", summaries_file)

        result = community_mod.get_community_summary(5)
        assert result == "This community is about Vietnamese geography."

    def test_falls_back_to_neo4j_when_not_in_jsonl(self, monkeypatch, tmp_path) -> None:
        # Reset the singleton
        monkeypatch.setattr(community_mod, "_summaries", None)

        # Empty JSONL file
        summaries_file = tmp_path / "summaries.jsonl"
        summaries_file.write_text("", encoding="utf-8")
        monkeypatch.setattr(community_mod, "_SUMMARIES_PATH", summaries_file)

        # Mock Neo4j to return a summary
        fake = _make_shared_session_factory([[{"summary": "Neo4j fallback summary"}]])
        monkeypatch.setattr(community_mod.neo4j_client, "session", fake)

        result = community_mod.get_community_summary(99)
        assert result == "Neo4j fallback summary"

    def test_returns_empty_when_not_found_anywhere(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(community_mod, "_summaries", None)

        summaries_file = tmp_path / "summaries.jsonl"
        summaries_file.write_text("", encoding="utf-8")
        monkeypatch.setattr(community_mod, "_SUMMARIES_PATH", summaries_file)

        fake = _make_shared_session_factory([[]])
        monkeypatch.setattr(community_mod.neo4j_client, "session", fake)

        result = community_mod.get_community_summary(999)
        assert result == ""

    def test_handles_integer_community_id_in_jsonl(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(community_mod, "_summaries", None)

        summaries_file = tmp_path / "summaries.jsonl"
        record = {
            "community_id": 7,
            "summary": "Integer ID community.",
            "member_count": 3,
            "top_entities": [],
            "entity_ids": [],
        }
        summaries_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
        monkeypatch.setattr(community_mod, "_SUMMARIES_PATH", summaries_file)

        result = community_mod.get_community_summary(7)
        assert result == "Integer ID community."


# ---------------------------------------------------------------------------
# Tests: retrieve_by_community
# ---------------------------------------------------------------------------


class TestRetrieveByCommunity:
    def test_returns_chunks_from_matched_communities(self, monkeypatch) -> None:
        # Call 1: entity fulltext search returns entities with community_id
        entity_results = [
            {"community_id": 1, "score": 5.0},
            {"community_id": 1, "score": 3.0},
            {"community_id": 2, "score": 2.0},
        ]
        # Call 2: chunk retrieval for community 1
        community1_chunks = [
            {
                "page_title": "Page A",
                "page_url": "http://a",
                "chunk_id": "c1",
                "chunk_text": "Text about community 1",
                "community_id": "community_00001",
            },
            {
                "page_title": "Page B",
                "page_url": "http://b",
                "chunk_id": "c2",
                "chunk_text": "More text about community 1",
                "community_id": "community_00001",
            },
        ]
        # Call 3: chunk retrieval for community 2
        community2_chunks = [
            {
                "page_title": "Page C",
                "page_url": "http://c",
                "chunk_id": "c3",
                "chunk_text": "Text about community 2",
                "community_id": "community_00002",
            },
        ]

        fake = _make_shared_session_factory([entity_results, community1_chunks, community2_chunks])
        monkeypatch.setattr(community_mod.neo4j_client, "session", fake)

        results = community_mod.retrieve_by_community("Hà Nội", top_k=5)

        assert len(results) == 3
        # Community 1 has higher accumulated score (5+3=8) so its chunks come first
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["score"] == 8.0
        assert results[2]["chunk_id"] == "c3"
        assert results[2]["score"] == 2.0

    def test_returns_empty_when_no_entities_match(self, monkeypatch) -> None:
        fake = _make_shared_session_factory([[]])
        monkeypatch.setattr(community_mod.neo4j_client, "session", fake)

        results = community_mod.retrieve_by_community("nonexistent query", top_k=5)
        assert results == []

    def test_deduplicates_chunks(self, monkeypatch) -> None:
        entity_results = [
            {"community_id": 1, "score": 5.0},
        ]
        # Same chunk_id appears twice in community results
        community_chunks = [
            {
                "page_title": "Page A",
                "page_url": "http://a",
                "chunk_id": "c1",
                "chunk_text": "Text",
                "community_id": "community_00001",
            },
            {
                "page_title": "Page A",
                "page_url": "http://a",
                "chunk_id": "c1",
                "chunk_text": "Text",
                "community_id": "community_00001",
            },
        ]

        fake = _make_shared_session_factory([entity_results, community_chunks])
        monkeypatch.setattr(community_mod.neo4j_client, "session", fake)

        results = community_mod.retrieve_by_community("test", top_k=5)
        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_respects_top_k(self, monkeypatch) -> None:
        entity_results = [
            {"community_id": 1, "score": 5.0},
        ]
        community_chunks = [
            {"page_title": f"P{i}", "page_url": f"http://{i}", "chunk_id": f"c{i}", "chunk_text": f"T{i}", "community_id": "community_00001"}
            for i in range(10)
        ]

        fake = _make_shared_session_factory([entity_results, community_chunks])
        monkeypatch.setattr(community_mod.neo4j_client, "session", fake)

        results = community_mod.retrieve_by_community("test", top_k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Tests: _load_summaries
# ---------------------------------------------------------------------------


class TestLoadSummaries:
    def test_handles_missing_file(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(community_mod, "_summaries", None)
        monkeypatch.setattr(community_mod, "_SUMMARIES_PATH", tmp_path / "nonexistent.jsonl")

        result = community_mod._load_summaries()
        assert result == {}

    def test_skips_invalid_community_id_format(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(community_mod, "_summaries", None)

        summaries_file = tmp_path / "summaries.jsonl"
        lines = [
            json.dumps({"community_id": "community_00001", "summary": "Valid"}),
            json.dumps({"community_id": "invalid_format", "summary": "Skipped"}),
            json.dumps({"community_id": "community_00003", "summary": "Also valid"}),
        ]
        summaries_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.setattr(community_mod, "_SUMMARIES_PATH", summaries_file)

        result = community_mod._load_summaries()
        assert 1 in result
        assert 3 in result
        assert len(result) == 2
