"""Tests for new agent tools: entity_neighborhood, path_search, majority_vote, run_agent_scaled."""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

import src.agent as agent_mod
from src.retrieve import QueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# Tests: _tool_entity_neighborhood
# ---------------------------------------------------------------------------


class TestToolEntityNeighborhood:
    def test_1hop_returns_entity_and_chunks(self, monkeypatch) -> None:
        fake_rows = [
            {
                "entity_name": "Hà Nội",
                "entity_type": "Location",
                "chunks": [
                    {"chunk_id": "c1", "page_title": "Hà Nội", "chunk_text": "Hà Nội là thủ đô..."},
                    {"chunk_id": "c2", "page_title": "Việt Nam", "chunk_text": "Thủ đô Hà Nội..."},
                ],
                "co_entities": [
                    {"name": "Việt Nam", "type": "Location"},
                    {"name": "Sông Hồng", "type": "Location"},
                ],
            }
        ]
        fake = _make_fake_session_factory(fake_rows)
        monkeypatch.setattr(agent_mod.neo4j_client, "session", fake)

        result = agent_mod._tool_entity_neighborhood("Hà Nội", hops=1)
        parsed = json.loads(result)

        assert parsed["entity"]["name"] == "Hà Nội"
        assert parsed["entity"]["type"] == "Location"
        assert len(parsed["chunks"]) == 2
        assert len(parsed["co_entities"]) == 2
        assert parsed["co_entities"][0]["name"] == "Việt Nam"

    def test_2hop_returns_extended_neighborhood(self, monkeypatch) -> None:
        fake_rows = [
            {
                "entity_name": "Hà Nội",
                "entity_type": "Location",
                "chunks": [
                    {"chunk_id": "c1", "page_title": "Hà Nội", "chunk_text": "text1"},
                ],
                "co_entities_hop1": [
                    {"name": "Việt Nam", "type": "Location"},
                ],
                "chunks_hop2": [
                    {"chunk_id": "c3", "page_title": "Việt Nam", "chunk_text": "text3"},
                ],
                "co_entities_hop2": [
                    {"name": "Đông Nam Á", "type": "Location"},
                ],
            }
        ]
        fake = _make_fake_session_factory(fake_rows)
        monkeypatch.setattr(agent_mod.neo4j_client, "session", fake)

        result = agent_mod._tool_entity_neighborhood("Hà Nội", hops=2)
        parsed = json.loads(result)

        assert parsed["entity"]["name"] == "Hà Nội"
        assert "co_entities_hop1" in parsed
        assert "chunks_hop2" in parsed
        assert "co_entities_hop2" in parsed

    def test_entity_not_found(self, monkeypatch) -> None:
        fake_rows = [{"entity_name": None, "entity_type": None, "chunks": [], "co_entities": []}]
        fake = _make_fake_session_factory(fake_rows)
        monkeypatch.setattr(agent_mod.neo4j_client, "session", fake)

        result = agent_mod._tool_entity_neighborhood("Nonexistent", hops=1)
        assert "not found" in result.lower()

    def test_empty_results(self, monkeypatch) -> None:
        fake = _make_fake_session_factory([])
        monkeypatch.setattr(agent_mod.neo4j_client, "session", fake)

        result = agent_mod._tool_entity_neighborhood("Empty", hops=1)
        assert "not found" in result.lower()

    def test_hops_clamped_to_range(self, monkeypatch) -> None:
        """Hops should be clamped to 1-3."""
        fake_rows = [
            {
                "entity_name": "Test",
                "entity_type": "Person",
                "chunks": [{"chunk_id": "c1", "page_title": "T", "chunk_text": "t"}],
                "co_entities": [{"name": "Other", "type": "Person"}],
            }
        ]
        fake = _make_fake_session_factory(fake_rows)
        monkeypatch.setattr(agent_mod.neo4j_client, "session", fake)

        # hops=0 should be clamped to 1 (uses 1-hop query)
        result = agent_mod._tool_entity_neighborhood("Test", hops=0)
        parsed = json.loads(result)
        assert "co_entities" in parsed  # 1-hop key

    def test_handles_exception(self, monkeypatch) -> None:
        @contextmanager
        def _failing_session():
            raise RuntimeError("Neo4j connection failed")
            yield  # noqa: unreachable

        monkeypatch.setattr(agent_mod.neo4j_client, "session", _failing_session)

        result = agent_mod._tool_entity_neighborhood("Test", hops=1)
        assert "Error" in result

    def test_filters_null_chunks_and_entities(self, monkeypatch) -> None:
        """Chunks/entities with null IDs/names should be filtered out."""
        fake_rows = [
            {
                "entity_name": "Test",
                "entity_type": "Person",
                "chunks": [
                    {"chunk_id": "c1", "page_title": "P", "chunk_text": "t"},
                    {"chunk_id": None, "page_title": None, "chunk_text": None},
                ],
                "co_entities": [
                    {"name": "Valid", "type": "Person"},
                    {"name": None, "type": None},
                ],
            }
        ]
        fake = _make_fake_session_factory(fake_rows)
        monkeypatch.setattr(agent_mod.neo4j_client, "session", fake)

        result = agent_mod._tool_entity_neighborhood("Test", hops=1)
        parsed = json.loads(result)

        assert len(parsed["chunks"]) == 1
        assert len(parsed["co_entities"]) == 1


# ---------------------------------------------------------------------------
# Tests: _tool_path_search
# ---------------------------------------------------------------------------


class TestToolPathSearch:
    def test_finds_path_between_entities(self, monkeypatch) -> None:
        fake_rows = [
            {
                "path_nodes": [
                    {"label": "Entity", "name": "Hà Nội", "type": "Location"},
                    {"label": "Chunk", "id": "c1", "text": "chunk text"},
                    {"label": "Entity", "name": "Việt Nam", "type": "Location"},
                ],
                "rel_types": ["MENTIONS", "MENTIONS"],
                "path_length": 2,
            }
        ]
        fake = _make_fake_session_factory(fake_rows)
        monkeypatch.setattr(agent_mod.neo4j_client, "session", fake)

        result = agent_mod._tool_path_search("Hà Nội", "Việt Nam", max_hops=3)
        parsed = json.loads(result)

        assert parsed["path_length"] == 2
        assert len(parsed["nodes"]) == 3
        assert len(parsed["relationships"]) == 2
        assert "Hà Nội" in parsed["path"]
        assert "Việt Nam" in parsed["path"]

    def test_no_path_found(self, monkeypatch) -> None:
        fake = _make_fake_session_factory([])
        monkeypatch.setattr(agent_mod.neo4j_client, "session", fake)

        result = agent_mod._tool_path_search("EntityA", "EntityB", max_hops=3)
        assert "No path found" in result

    def test_max_hops_clamped(self, monkeypatch) -> None:
        """max_hops should be clamped to 1-5."""
        fake_rows = [
            {
                "path_nodes": [
                    {"label": "Entity", "name": "A", "type": "Person"},
                    {"label": "Entity", "name": "B", "type": "Person"},
                ],
                "rel_types": ["MENTIONS"],
                "path_length": 1,
            }
        ]
        fake = _make_fake_session_factory(fake_rows)
        monkeypatch.setattr(agent_mod.neo4j_client, "session", fake)

        # max_hops=10 should be clamped to 5
        result = agent_mod._tool_path_search("A", "B", max_hops=10)
        parsed = json.loads(result)
        assert parsed["path_length"] == 1

    def test_handles_exception(self, monkeypatch) -> None:
        @contextmanager
        def _failing_session():
            raise RuntimeError("Connection error")
            yield  # noqa: unreachable

        monkeypatch.setattr(agent_mod.neo4j_client, "session", _failing_session)

        result = agent_mod._tool_path_search("A", "B", max_hops=3)
        assert "Error" in result

    def test_path_with_page_node(self, monkeypatch) -> None:
        fake_rows = [
            {
                "path_nodes": [
                    {"label": "Entity", "name": "A", "type": "Person"},
                    {"label": "Page", "title": "Page Title", "url": "http://x"},
                    {"label": "Entity", "name": "B", "type": "Person"},
                ],
                "rel_types": ["HAS_CHUNK", "MENTIONS"],
                "path_length": 2,
            }
        ]
        fake = _make_fake_session_factory(fake_rows)
        monkeypatch.setattr(agent_mod.neo4j_client, "session", fake)

        result = agent_mod._tool_path_search("A", "B", max_hops=3)
        parsed = json.loads(result)

        assert "Page Title" in parsed["path"]


# ---------------------------------------------------------------------------
# Tests: _majority_vote
# ---------------------------------------------------------------------------


class TestMajorityVote:
    def test_clear_majority(self) -> None:
        """When 3 out of 5 results agree, the majority answer wins."""
        results = [
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c1"}]),
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c1"}, {"chunk_id": "c2"}]),
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c3"}]),
            QueryResult(answer="Sài Gòn", citations=[{"chunk_id": "c4"}]),
            QueryResult(answer="Đà Nẵng", citations=[{"chunk_id": "c5"}]),
        ]

        winner = agent_mod._majority_vote(results)
        assert "Hà Nội" in winner.answer

    def test_tie_broken_by_citations(self) -> None:
        """When groups are tied in size, the one with more citations wins."""
        results = [
            QueryResult(answer="Answer A", citations=[{"chunk_id": "c1"}]),
            QueryResult(answer="Answer B", citations=[{"chunk_id": "c2"}, {"chunk_id": "c3"}, {"chunk_id": "c4"}]),
        ]

        winner = agent_mod._majority_vote(results)
        # Both groups have size 1, so tie-break by max citations
        assert winner.answer == "Answer B"

    def test_containment_groups_similar_answers(self) -> None:
        """Answers where one contains the other should be grouped together."""
        results = [
            QueryResult(answer="Hà Nội là thủ đô của Việt Nam", citations=[{"chunk_id": "c1"}]),
            QueryResult(answer="Hà Nội là thủ đô của Việt Nam, nằm ở miền Bắc", citations=[{"chunk_id": "c2"}]),
            QueryResult(answer="Sài Gòn", citations=[{"chunk_id": "c3"}]),
        ]

        winner = agent_mod._majority_vote(results)
        # The two Hà Nội answers should be grouped (containment), forming majority
        assert "Hà Nội" in winner.answer

    def test_single_result(self) -> None:
        """Single result should be returned as-is."""
        result = QueryResult(answer="Only answer", citations=[{"chunk_id": "c1"}])
        winner = agent_mod._majority_vote([result])
        assert winner.answer == "Only answer"
        assert winner is result

    def test_empty_results(self) -> None:
        """Empty results list should return a default answer."""
        winner = agent_mod._majority_vote([])
        assert "Không tìm thấy" in winner.answer
        assert winner.citations == []

    def test_normalized_comparison(self) -> None:
        """Answers differing only in case/whitespace/punctuation should be grouped."""
        results = [
            QueryResult(answer="Hà Nội.", citations=[{"chunk_id": "c1"}]),
            QueryResult(answer="hà nội", citations=[{"chunk_id": "c2"}]),
            QueryResult(answer="Sài Gòn", citations=[{"chunk_id": "c3"}]),
        ]

        winner = agent_mod._majority_vote(results)
        # Both "Hà Nội." and "hà nội" normalize to same string
        assert "hà nội" in winner.answer.lower()

    def test_winner_has_most_citations_in_group(self) -> None:
        """Within the winning group, the result with most citations is selected."""
        results = [
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c1"}]),
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c2"}, {"chunk_id": "c3"}, {"chunk_id": "c4"}]),
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c5"}, {"chunk_id": "c6"}]),
        ]

        winner = agent_mod._majority_vote(results)
        assert len(winner.citations) == 3


# ---------------------------------------------------------------------------
# Tests: _answers_similar
# ---------------------------------------------------------------------------


class TestAnswersSimilar:
    def test_exact_match(self) -> None:
        assert agent_mod._answers_similar("Hà Nội", "Hà Nội") is True

    def test_case_insensitive(self) -> None:
        assert agent_mod._answers_similar("Hà Nội", "hà nội") is True

    def test_trailing_punctuation_ignored(self) -> None:
        assert agent_mod._answers_similar("Hà Nội.", "Hà Nội") is True

    def test_containment(self) -> None:
        assert agent_mod._answers_similar(
            "Hà Nội là thủ đô",
            "Hà Nội là thủ đô của Việt Nam"
        ) is True

    def test_different_answers(self) -> None:
        assert agent_mod._answers_similar("Hà Nội", "Sài Gòn") is False

    def test_short_strings_no_containment(self) -> None:
        """Short strings (<=10 chars) don't use containment check."""
        assert agent_mod._answers_similar("abc", "abcdef") is False


# ---------------------------------------------------------------------------
# Tests: run_agent_scaled
# ---------------------------------------------------------------------------


class TestRunAgentScaled:
    def test_n1_delegates_to_agent_query(self, monkeypatch) -> None:
        """With n_trajectories=1, should just call agent_query."""
        called = [False]

        def _fake_agent_query(question, top_k=4):
            called[0] = True
            return QueryResult(answer="Direct answer", citations=[])

        monkeypatch.setattr(agent_mod, "agent_query", _fake_agent_query)

        result = agent_mod.run_agent_scaled("Test question", n_trajectories=1)
        assert called[0] is True
        assert result.answer == "Direct answer"

    def test_n_from_settings_when_none(self, monkeypatch) -> None:
        """When n_trajectories is None, uses settings.agent_n_trajectories."""
        monkeypatch.setattr(agent_mod.settings, "agent_n_trajectories", 1)

        called = [False]

        def _fake_agent_query(question, top_k=4):
            called[0] = True
            return QueryResult(answer="Settings answer", citations=[])

        monkeypatch.setattr(agent_mod, "agent_query", _fake_agent_query)

        result = agent_mod.run_agent_scaled("Test question", n_trajectories=None)
        assert called[0] is True

    def test_multiple_trajectories_uses_majority_vote(self, monkeypatch) -> None:
        """With n>1, runs multiple trajectories and uses majority vote."""
        monkeypatch.setattr(agent_mod.settings, "agent_temperature_scaled", 0.7)

        trajectory_results = [
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c1"}]),
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c2"}]),
            QueryResult(answer="Sài Gòn", citations=[{"chunk_id": "c3"}]),
        ]
        call_idx = [0]

        def _fake_run_trajectory(question, tid, temperature):
            idx = call_idx[0]
            call_idx[0] += 1
            return trajectory_results[idx % len(trajectory_results)]

        monkeypatch.setattr(agent_mod, "_run_trajectory", _fake_run_trajectory)

        result = agent_mod.run_agent_scaled("Thủ đô Việt Nam?", n_trajectories=3)

        assert "Hà Nội" in result.answer
        assert result.retrieval_tier == "scaled_3"

    def test_all_trajectories_fail(self, monkeypatch) -> None:
        """If all trajectories raise exceptions, returns fallback answer."""
        monkeypatch.setattr(agent_mod.settings, "agent_temperature_scaled", 0.7)

        def _failing_trajectory(question, tid, temperature):
            raise RuntimeError("Trajectory failed")

        monkeypatch.setattr(agent_mod, "_run_trajectory", _failing_trajectory)

        result = agent_mod.run_agent_scaled("Test", n_trajectories=2)
        assert "Không tìm thấy" in result.answer
