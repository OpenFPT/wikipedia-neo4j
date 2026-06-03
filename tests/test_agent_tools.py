"""Tests for new agent tools: entity_neighborhood, path_search, majority_vote, run_agent_scaled."""

from __future__ import annotations

import json
from contextlib import contextmanager

import src.orchestration.agent as agent_mod
import src.orchestration.agent_loop as agent_loop_mod
import src.orchestration.voting as voting_mod
from src.retrieval.fusion import QueryResult
from src.orchestration.agent_loop import _tool_entity_neighborhood, _tool_path_search
from src.orchestration.voting import run_agent_scaled, _majority_vote, _answers_similar


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


class _FakeNeo4jClient:
    """Fake neo4j_client that returns a context-managed session."""

    def __init__(self, results: list[dict]):
        self._results = results

    @contextmanager
    def session(self):
        yield _FakeSession(self._results)


class _FailingNeo4jClient:
    """Fake neo4j_client whose session() always raises."""

    def __init__(self, exc: Exception):
        self._exc = exc

    @contextmanager
    def session(self):
        raise self._exc
        yield  # noqa: RET503


def _make_fake_session_factory(results: list[dict]):
    @contextmanager
    def _session():
        yield _FakeSession(results)

    return _session


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
        monkeypatch.setattr(agent_loop_mod, "neo4j_client", _FakeNeo4jClient(fake_rows))

        result = _tool_entity_neighborhood("Hà Nội", hops=1)
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
        monkeypatch.setattr(agent_loop_mod, "neo4j_client", _FakeNeo4jClient(fake_rows))

        result = _tool_entity_neighborhood("Hà Nội", hops=2)
        parsed = json.loads(result)

        assert parsed["entity"]["name"] == "Hà Nội"
        assert "co_entities_hop1" in parsed
        assert "chunks_hop2" in parsed
        assert "co_entities_hop2" in parsed

    def test_entity_not_found(self, monkeypatch) -> None:
        fake_rows = [{"entity_name": None, "entity_type": None, "chunks": [], "co_entities": []}]
        monkeypatch.setattr(agent_loop_mod, "neo4j_client", _FakeNeo4jClient(fake_rows))

        result = _tool_entity_neighborhood("Nonexistent", hops=1)
        assert "not found" in result.lower()

    def test_empty_results(self, monkeypatch) -> None:
        monkeypatch.setattr(agent_loop_mod, "neo4j_client", _FakeNeo4jClient([]))

        result = _tool_entity_neighborhood("Empty", hops=1)
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
        monkeypatch.setattr(agent_loop_mod, "neo4j_client", _FakeNeo4jClient(fake_rows))

        # hops=0 should be clamped to 1 (uses 1-hop query)
        result = _tool_entity_neighborhood("Test", hops=0)
        parsed = json.loads(result)
        assert "co_entities" in parsed  # 1-hop key

    def test_handles_exception(self, monkeypatch) -> None:
        monkeypatch.setattr(agent_loop_mod, "neo4j_client", _FailingNeo4jClient(RuntimeError("Neo4j connection failed")))

        result = _tool_entity_neighborhood("Test", hops=1)
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
        monkeypatch.setattr(agent_loop_mod, "neo4j_client", _FakeNeo4jClient(fake_rows))

        result = _tool_entity_neighborhood("Test", hops=1)
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
        monkeypatch.setattr(agent_loop_mod, "neo4j_client", _FakeNeo4jClient(fake_rows))

        result = _tool_path_search("Hà Nội", "Việt Nam", max_hops=3)
        parsed = json.loads(result)

        assert parsed["path_length"] == 2
        assert len(parsed["nodes"]) == 3
        assert len(parsed["relationships"]) == 2
        assert "Hà Nội" in parsed["path"]
        assert "Việt Nam" in parsed["path"]

    def test_no_path_found(self, monkeypatch) -> None:
        monkeypatch.setattr(agent_loop_mod, "neo4j_client", _FakeNeo4jClient([]))

        result = _tool_path_search("EntityA", "EntityB", max_hops=3)
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
        monkeypatch.setattr(agent_loop_mod, "neo4j_client", _FakeNeo4jClient(fake_rows))

        # max_hops=10 should be clamped to 5
        result = _tool_path_search("A", "B", max_hops=10)
        parsed = json.loads(result)
        assert parsed["path_length"] == 1

    def test_handles_exception(self, monkeypatch) -> None:
        monkeypatch.setattr(agent_loop_mod, "neo4j_client", _FailingNeo4jClient(RuntimeError("Connection error")))

        result = _tool_path_search("A", "B", max_hops=3)
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
        monkeypatch.setattr(agent_loop_mod, "neo4j_client", _FakeNeo4jClient(fake_rows))

        result = _tool_path_search("A", "B", max_hops=3)
        parsed = json.loads(result)

        assert "Page Title" in parsed["path"]


# ---------------------------------------------------------------------------
# Tests: _majority_vote
# ---------------------------------------------------------------------------


class TestMajorityVote:
    def test_clear_majority(self) -> None:
        results = [
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c1"}]),
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c1"}, {"chunk_id": "c2"}]),
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c3"}]),
            QueryResult(answer="Sài Gòn", citations=[{"chunk_id": "c4"}]),
            QueryResult(answer="Đà Nẵng", citations=[{"chunk_id": "c5"}]),
        ]

        winner = _majority_vote(results)
        assert "Hà Nội" in winner.answer

    def test_tie_broken_by_citations(self) -> None:
        results = [
            QueryResult(answer="Answer A", citations=[{"chunk_id": "c1"}]),
            QueryResult(answer="Answer B", citations=[{"chunk_id": "c2"}, {"chunk_id": "c3"}, {"chunk_id": "c4"}]),
        ]

        winner = _majority_vote(results)
        assert winner.answer == "Answer B"

    def test_containment_groups_similar_answers(self) -> None:
        results = [
            QueryResult(answer="Hà Nội là thủ đô của Việt Nam", citations=[{"chunk_id": "c1"}]),
            QueryResult(answer="Hà Nội là thủ đô của Việt Nam, nằm ở miền Bắc", citations=[{"chunk_id": "c2"}]),
            QueryResult(answer="Sài Gòn", citations=[{"chunk_id": "c3"}]),
        ]

        winner = _majority_vote(results)
        assert "Hà Nội" in winner.answer

    def test_single_result(self) -> None:
        result = QueryResult(answer="Only answer", citations=[{"chunk_id": "c1"}])
        winner = _majority_vote([result])
        assert winner.answer == "Only answer"
        assert winner is result

    def test_empty_results(self) -> None:
        winner = _majority_vote([])
        assert "Không tìm thấy" in winner.answer
        assert winner.citations == []

    def test_normalized_comparison(self) -> None:
        results = [
            QueryResult(answer="Hà Nội.", citations=[{"chunk_id": "c1"}]),
            QueryResult(answer="hà nội", citations=[{"chunk_id": "c2"}]),
            QueryResult(answer="Sài Gòn", citations=[{"chunk_id": "c3"}]),
        ]

        winner = _majority_vote(results)
        assert "hà nội" in winner.answer.lower()

    def test_winner_has_most_citations_in_group(self) -> None:
        results = [
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c1"}]),
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c2"}, {"chunk_id": "c3"}, {"chunk_id": "c4"}]),
            QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c5"}, {"chunk_id": "c6"}]),
        ]

        winner = _majority_vote(results)
        assert len(winner.citations) == 3


# ---------------------------------------------------------------------------
# Tests: _answers_similar
# ---------------------------------------------------------------------------


class TestAnswersSimilar:
    def test_exact_match(self) -> None:
        assert _answers_similar("Hà Nội", "Hà Nội") is True

    def test_case_insensitive(self) -> None:
        assert _answers_similar("Hà Nội", "hà nội") is True

    def test_trailing_punctuation_ignored(self) -> None:
        assert _answers_similar("Hà Nội.", "Hà Nội") is True

    def test_containment(self) -> None:
        assert _answers_similar(
            "Hà Nội là thủ đô",
            "Hà Nội là thủ đô của Việt Nam"
        ) is True

    def test_different_answers(self) -> None:
        assert _answers_similar("Hà Nội", "Sài Gòn") is False

    def test_short_strings_no_containment(self) -> None:
        assert _answers_similar("abc", "abcdef") is False


# ---------------------------------------------------------------------------
# Tests: run_agent_scaled
# ---------------------------------------------------------------------------


class TestRunAgentScaled:
    def test_n1_delegates_to_agent_query(self, monkeypatch) -> None:
        called = [False]

        def _fake_agent_query(question, top_k=4):
            called[0] = True
            return QueryResult(answer="Direct answer", citations=[])

        monkeypatch.setattr(voting_mod, "agent_query", _fake_agent_query)

        result = run_agent_scaled("Test question", n_trajectories=1)
        assert called[0] is True
        assert result.answer == "Direct answer"

    def test_n_from_settings_when_none(self, monkeypatch) -> None:
        monkeypatch.setattr(voting_mod.settings, "agent_n_trajectories", 1)

        called = [False]

        def _fake_agent_query(question, top_k=4):
            called[0] = True
            return QueryResult(answer="Settings answer", citations=[])

        monkeypatch.setattr(voting_mod, "agent_query", _fake_agent_query)

        run_agent_scaled("Test question", n_trajectories=None)
        assert called[0] is True

    def test_multiple_trajectories_uses_majority_vote(self, monkeypatch) -> None:
        monkeypatch.setattr(voting_mod.settings, "agent_temperature_scaled", 0.7)

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

        monkeypatch.setattr(voting_mod, "_run_trajectory", _fake_run_trajectory)

        result = run_agent_scaled("Thủ đô Việt Nam?", n_trajectories=3)

        assert "Hà Nội" in result.answer
        assert result.retrieval_tier == "scaled_3"

    def test_all_trajectories_fail(self, monkeypatch) -> None:
        monkeypatch.setattr(voting_mod.settings, "agent_temperature_scaled", 0.7)

        def _failing_trajectory(question, tid, temperature):
            raise RuntimeError("Trajectory failed")

        monkeypatch.setattr(voting_mod, "_run_trajectory", _failing_trajectory)

        result = run_agent_scaled("Test", n_trajectories=2)
        assert "Không tìm thấy" in result.answer


# ---------------------------------------------------------------------------
# Tests: _tool_kg_query
# ---------------------------------------------------------------------------


class TestToolKgQuery:
    def test_returns_results(self, monkeypatch) -> None:
        monkeypatch.setattr(
            agent_mod.neo4j_client, "session",
            _make_fake_session_factory([{"name": "Hà Nội", "type": "Location"}]),
        )
        result = agent_mod._tool_kg_query(
            "MATCH (e:Entity) RETURN e.name AS page_title, e.type AS page_url, "
            "e.name AS chunk_id, e.name AS chunk_text, 1.0 AS score LIMIT $top_k"
        )
        data = json.loads(result)
        assert data[0]["name"] == "Hà Nội"

    def test_no_results(self, monkeypatch) -> None:
        monkeypatch.setattr(
            agent_mod.neo4j_client, "session",
            _make_fake_session_factory([]),
        )
        result = agent_mod._tool_kg_query(
            "MATCH (e:Entity) RETURN e.name AS page_title, e.type AS page_url, "
            "e.name AS chunk_id, e.name AS chunk_text, 1.0 AS score LIMIT $top_k"
        )
        assert "No results" in result

    def test_rejects_write_query(self) -> None:
        result = agent_mod._tool_kg_query("CREATE (n:Node {name: 'x'})")
        assert "Error" in result

    def test_handles_db_exception(self, monkeypatch) -> None:
        @contextmanager
        def _failing_session():
            raise RuntimeError("connection lost")
            yield  # noqa: unreachable

        monkeypatch.setattr(agent_mod.neo4j_client, "session", _failing_session)
        result = agent_mod._tool_kg_query(
            "MATCH (e:Entity) RETURN e.name AS page_title, e.type AS page_url, "
            "e.name AS chunk_id, e.name AS chunk_text, 1.0 AS score LIMIT $top_k"
        )
        assert "Error" in result


# ---------------------------------------------------------------------------
# Tests: _tool_text_search
# ---------------------------------------------------------------------------


class TestToolTextSearch:
    def test_returns_results(self, monkeypatch) -> None:
        monkeypatch.setattr(
            agent_mod.neo4j_client, "session",
            _make_fake_session_factory([
                {"page_title": "Test", "page_url": "http://x", "chunk_id": "c1", "chunk_text": "text", "score": 0.9}
            ]),
        )
        result = agent_mod._tool_text_search("test query")
        data = json.loads(result)
        assert data[0]["page_title"] == "Test"

    def test_no_results(self, monkeypatch) -> None:
        monkeypatch.setattr(
            agent_mod.neo4j_client, "session",
            _make_fake_session_factory([]),
        )
        result = agent_mod._tool_text_search("nothing")
        assert "No results" in result

    def test_handles_exception(self, monkeypatch) -> None:
        @contextmanager
        def _failing_session():
            raise RuntimeError("timeout")
            yield  # noqa: unreachable

        monkeypatch.setattr(agent_mod.neo4j_client, "session", _failing_session)
        result = agent_mod._tool_text_search("test")
        assert "Error" in result


# ---------------------------------------------------------------------------
# Tests: _tool_get_passage
# ---------------------------------------------------------------------------


class TestToolGetPassage:
    def test_returns_passage(self, monkeypatch) -> None:
        monkeypatch.setattr(
            agent_mod.neo4j_client, "session",
            _make_fake_session_factory([
                {"page_title": "Page1", "page_url": "http://x", "chunk_text": "Some content"}
            ]),
        )
        result = agent_mod._tool_get_passage("c1")
        data = json.loads(result)
        assert data["chunk_text"] == "Some content"

    def test_chunk_not_found(self, monkeypatch) -> None:
        monkeypatch.setattr(
            agent_mod.neo4j_client, "session",
            _make_fake_session_factory([]),
        )
        result = agent_mod._tool_get_passage("missing")
        assert "not found" in result

    def test_handles_exception(self, monkeypatch) -> None:
        @contextmanager
        def _failing_session():
            raise RuntimeError("db error")
            yield  # noqa: unreachable

        monkeypatch.setattr(agent_mod.neo4j_client, "session", _failing_session)
        result = agent_mod._tool_get_passage("c1")
        assert "Error" in result


# ---------------------------------------------------------------------------
# Tests: _check_sufficiency
# ---------------------------------------------------------------------------


class TestCheckSufficiencyExtra:
    def test_no_valid_observations(self) -> None:
        is_suff, conf = agent_mod._check_sufficiency(
            ["Error: something", "No results found.", ""], "question"
        )
        assert is_suff is False
        assert conf == 0.0

    def test_sufficient_with_chunks(self) -> None:
        obs = [
            json.dumps([
                {"chunk_id": "c1", "text": "a"},
                {"chunk_id": "c2", "text": "b"},
                {"chunk_id": "c3", "text": "c"},
            ]),
        ]
        is_suff, conf = agent_mod._check_sufficiency(obs, "question")
        assert is_suff is True
        assert conf >= 0.5

    def test_dict_observation_with_chunk_id(self) -> None:
        obs = [json.dumps({"chunk_id": "c1", "text": "data"})]
        is_suff, conf = agent_mod._check_sufficiency(obs, "question")
        assert conf > 0.0

    def test_non_json_observations(self) -> None:
        obs = ["Some plain text observation that is valid"]
        is_suff, conf = agent_mod._check_sufficiency(obs, "question")
        assert conf > 0.0


# ---------------------------------------------------------------------------
# Tests: _parse_agent_response
# ---------------------------------------------------------------------------


class TestParseAgentResponseExtra:
    def test_parses_code_fenced_json(self) -> None:
        raw = '```json\n{"action": "text_search", "action_input": "test"}\n```'
        result = agent_mod._parse_agent_response(raw)
        assert result["action"] == "text_search"

    def test_returns_none_for_no_json(self) -> None:
        result = agent_mod._parse_agent_response("This is just plain text with no JSON")
        assert result is None

    def test_returns_none_for_invalid_json(self) -> None:
        result = agent_mod._parse_agent_response("{invalid json content here}")
        assert result is None
