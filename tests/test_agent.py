"""Tests for ReAct agent loop and tools."""

from __future__ import annotations

import json

import src.orchestration.agent as agent
from src.retrieval.hybrid import QueryResult


class TestParseAgentResponse:
    def test_valid_action(self) -> None:
        raw = '{"thought": "I need to search", "action": "text_search", "action_input": {"query": "Hà Nội"}}'
        result = agent._parse_agent_response(raw)
        assert result is not None
        assert result["action"] == "text_search"

    def test_valid_final_answer(self) -> None:
        raw = '{"thought": "I found it", "final_answer": "Hà Nội là thủ đô của Việt Nam"}'
        result = agent._parse_agent_response(raw)
        assert result is not None
        assert result["final_answer"] == "Hà Nội là thủ đô của Việt Nam"

    def test_code_fence_stripped(self) -> None:
        raw = '```json\n{"thought": "x", "action": "kg_schema", "action_input": {}}\n```'
        result = agent._parse_agent_response(raw)
        assert result is not None
        assert result["action"] == "kg_schema"

    def test_invalid_json_returns_none(self) -> None:
        assert agent._parse_agent_response("not json at all") is None

    def test_embedded_json(self) -> None:
        raw = 'Here is my response: {"thought": "ok", "final_answer": "done"} end'
        result = agent._parse_agent_response(raw)
        assert result is not None
        assert result["final_answer"] == "done"


class TestToolKgSchema:
    def test_returns_schema_string(self) -> None:
        result = agent._tool_kg_schema()
        assert "Page" in result
        assert "Chunk" in result
        assert "Entity" in result


class TestToolKgQuery:
    def test_rejects_write_cypher(self) -> None:
        result = agent._tool_kg_query("CREATE (n:Test {name: 'bad'})")
        assert "Error" in result
        assert "not read-only" in result

    def test_rejects_empty_cypher(self) -> None:
        result = agent._tool_kg_query("")
        assert "Error" in result

    def test_valid_cypher_executes(self, monkeypatch) -> None:
        class _FakeSession:
            def run(self, cypher, **params):
                return [{"page_title": "Test", "page_url": "http://x", "chunk_id": "c1", "chunk_text": "hello", "score": 1.0}]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        from contextlib import contextmanager

        @contextmanager
        def _fake_session():
            yield _FakeSession()

        monkeypatch.setattr(agent.neo4j_client, "session", _fake_session)
        result = agent._tool_kg_query(
            "MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk) "
            "RETURN p.title AS page_title, p.url AS page_url, c.id AS chunk_id, c.text AS chunk_text, 1.0 AS score LIMIT 5"
        )
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["page_title"] == "Test"


class TestToolTextSearch:
    def test_returns_results(self, monkeypatch) -> None:
        class _FakeSession:
            def run(self, cypher, **params):
                return [{"page_title": "Hà Nội", "page_url": "http://x", "chunk_id": "c1", "chunk_text": "text", "score": 2.0}]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        from contextlib import contextmanager

        @contextmanager
        def _fake_session():
            yield _FakeSession()

        monkeypatch.setattr(agent.neo4j_client, "session", _fake_session)
        result = agent._tool_text_search("Hà Nội")
        parsed = json.loads(result)
        assert parsed[0]["page_title"] == "Hà Nội"

    def test_empty_results(self, monkeypatch) -> None:
        class _FakeSession:
            def run(self, cypher, **params):
                return []

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        from contextlib import contextmanager

        @contextmanager
        def _fake_session():
            yield _FakeSession()

        monkeypatch.setattr(agent.neo4j_client, "session", _fake_session)
        result = agent._tool_text_search("nonexistent")
        assert result == "No results found."


class TestToolGetPassage:
    def test_returns_passage(self, monkeypatch) -> None:
        class _FakeSession:
            def run(self, cypher, **params):
                return [{"page_title": "Test", "page_url": "http://x", "chunk_text": "Full passage text here"}]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        from contextlib import contextmanager

        @contextmanager
        def _fake_session():
            yield _FakeSession()

        monkeypatch.setattr(agent.neo4j_client, "session", _fake_session)
        result = agent._tool_get_passage("c1")
        parsed = json.loads(result)
        assert parsed["chunk_text"] == "Full passage text here"

    def test_not_found(self, monkeypatch) -> None:
        class _FakeSession:
            def run(self, cypher, **params):
                return []

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        from contextlib import contextmanager

        @contextmanager
        def _fake_session():
            yield _FakeSession()

        monkeypatch.setattr(agent.neo4j_client, "session", _fake_session)
        result = agent._tool_get_passage("nonexistent")
        assert result == "Chunk not found."


class TestAgentQuery:
    def test_converges_with_final_answer(self, monkeypatch) -> None:
        call_count = [0]

        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            call_count[0] += 1
            # Call 1: complexity detection (returns low complexity)
            if call_count[0] == 1:
                return json.dumps({"complexity": 1})
            # Call 2: agent loop - tool action
            if call_count[0] == 2:
                return json.dumps({
                    "thought": "Let me search for the answer",
                    "action": "text_search",
                    "action_input": {"query": "Việt Nam thủ đô"},
                })
            # Call 3+: agent loop - final answer
            return json.dumps({
                "thought": "I found the answer",
                "final_answer": "Hà Nội là thủ đô của Việt Nam.",
            })

        class _FakeSession:
            def run(self, cypher, **params):
                return [{"page_title": "Hà Nội", "page_url": "http://x", "chunk_id": "c1", "chunk_text": "Hà Nội là thủ đô", "score": 1.0}]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        from contextlib import contextmanager

        @contextmanager
        def _fake_session():
            yield _FakeSession()

        monkeypatch.setattr(agent.neo4j_client, "session", _fake_session)

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent.agent_query("Thủ đô của Việt Nam là gì?")
        assert result.answer == "Hà Nội là thủ đô của Việt Nam."
        assert len(result.citations) == 1
        assert result.citations[0]["chunk_id"] == "c1"

    def test_fallback_on_unparseable(self, monkeypatch) -> None:
        call_count = [0]

        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            call_count[0] += 1
            # Call 1: complexity detection
            if call_count[0] == 1:
                return json.dumps({"complexity": 1})
            return "I don't know how to respond in JSON"

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent.agent_query("test question")
        assert "Không tìm thấy" in result.answer

    def test_unknown_tool_handled(self, monkeypatch) -> None:
        call_count = [0]

        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            call_count[0] += 1
            # Call 1: complexity detection
            if call_count[0] == 1:
                return json.dumps({"complexity": 1})
            # Call 2: agent loop - bad tool
            if call_count[0] == 2:
                return json.dumps({
                    "thought": "try bad tool",
                    "action": "bad_tool",
                    "action_input": {},
                })
            return json.dumps({
                "thought": "ok",
                "final_answer": "fallback answer",
            })

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent.agent_query("test")
        assert result.answer == "fallback answer"


class TestCheckSufficiency:
    def test_empty_observations(self):
        is_suff, conf = agent._check_sufficiency([], "question")
        assert is_suff is False
        assert conf == 0.0

    def test_only_errors(self):
        obs = ["Error: connection failed", "No results found."]
        is_suff, conf = agent._check_sufficiency(obs, "question")
        assert is_suff is False
        assert conf == 0.0

    def test_sufficient_with_chunks(self):
        chunks = [
            {"chunk_id": "c1", "chunk_text": "text1"},
            {"chunk_id": "c2", "chunk_text": "text2"},
            {"chunk_id": "c3", "chunk_text": "text3"},
        ]
        obs = [json.dumps(chunks)]
        is_suff, conf = agent._check_sufficiency(obs, "question")
        assert is_suff is True
        assert conf > 0.5

    def test_partial_evidence(self):
        chunks = [{"chunk_id": "c1", "chunk_text": "text1"}]
        obs = [json.dumps(chunks)]
        is_suff, conf = agent._check_sufficiency(obs, "question")
        assert conf > 0.0

    def test_non_json_valid_observation(self):
        obs = ["Some plain text observation that is valid"]
        is_suff, conf = agent._check_sufficiency(obs, "question")
        assert conf > 0.0


class TestSynthesizeFromObservations:
    def test_with_valid_observations(self):
        obs = ["Hà Nội là thủ đô", "Error: failed", "Dân số 8 triệu"]
        citations = [{"page_title": "Hà Nội", "chunk_id": "c1"}]
        result = agent._synthesize_from_observations(obs, citations)
        assert "Dựa trên thông tin" in result.answer
        assert result.citations == citations

    def test_with_no_valid_observations(self):
        obs = ["Error: failed", "No results found."]
        result = agent._synthesize_from_observations(obs, [])
        assert "Không tìm thấy" in result.answer

    def test_empty_observations(self):
        result = agent._synthesize_from_observations([], [])
        assert "Không tìm thấy" in result.answer


class TestAnswersSimilar:
    def test_exact_match(self):
        assert agent._answers_similar("Hà Nội", "Hà Nội") is True

    def test_case_insensitive(self):
        assert agent._answers_similar("hà nội", "Hà Nội") is True

    def test_trailing_punctuation(self):
        assert agent._answers_similar("Hà Nội.", "Hà Nội") is True

    def test_containment(self):
        assert agent._answers_similar(
            "Hà Nội là thủ đô",
            "Hà Nội là thủ đô của Việt Nam",
        ) is True

    def test_different_answers(self):
        assert agent._answers_similar("Hà Nội", "Sài Gòn") is False

    def test_short_strings_no_containment(self):
        assert agent._answers_similar("abc", "abcdef") is False


class TestMajorityVote:
    def test_empty_results(self):
        result = agent._majority_vote([])
        assert "Không tìm thấy" in result.answer

    def test_single_result(self):
        r = QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c1"}])
        result = agent._majority_vote([r])
        assert result.answer == "Hà Nội"

    def test_majority_wins(self):
        r1 = QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c1"}])
        r2 = QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c1"}, {"chunk_id": "c2"}])
        r3 = QueryResult(answer="Sài Gòn", citations=[{"chunk_id": "c3"}])
        result = agent._majority_vote([r1, r2, r3])
        assert result.answer == "Hà Nội"
        assert len(result.citations) == 2

    def test_tie_breaks_by_citations(self):
        r1 = QueryResult(answer="A answer long enough", citations=[{"chunk_id": "c1"}])
        r2 = QueryResult(answer="B answer long enough", citations=[{"chunk_id": "c2"}, {"chunk_id": "c3"}])
        result = agent._majority_vote([r1, r2])
        assert len(result.citations) == 2


class TestRunAgentScaled:
    def test_single_trajectory_falls_back(self, monkeypatch):
        call_count = [0]

        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"complexity": 1})
            if call_count[0] == 2:
                return json.dumps({
                    "thought": "search",
                    "action": "text_search",
                    "action_input": {"query": "test"},
                })
            return json.dumps({
                "thought": "done",
                "final_answer": "answer from single",
            })

        from contextlib import contextmanager

        class _FakeSession:
            def run(self, cypher, **params):
                return [{"page_title": "T", "page_url": "u", "chunk_id": "c1", "chunk_text": "t", "score": 1.0}]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        @contextmanager
        def _fake_session():
            yield _FakeSession()

        monkeypatch.setattr(agent.neo4j_client, "session", _fake_session)

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent.run_agent_scaled("test", n_trajectories=1)
        assert result.answer == "answer from single"

    def test_multi_trajectory_uses_voting(self, monkeypatch):
        def _fake_trajectory(question, tid, temperature):
            if tid == 0:
                return QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c1"}])
            elif tid == 1:
                return QueryResult(answer="Hà Nội", citations=[{"chunk_id": "c1"}, {"chunk_id": "c2"}])
            else:
                return QueryResult(answer="Sài Gòn", citations=[{"chunk_id": "c3"}])

        monkeypatch.setattr(agent, "_run_trajectory", _fake_trajectory)

        result = agent.run_agent_scaled("Thủ đô?", n_trajectories=3)
        assert "Hà Nội" in result.answer
        assert result.retrieval_tier == "scaled_3"


class TestNormalizeAnswerAgent:
    def test_strips_trailing_period(self):
        assert agent._normalize_answer("Hà Nội.") == "hà nội"

    def test_strips_whitespace(self):
        assert agent._normalize_answer("  Hà Nội  ") == "hà nội"

    def test_lowercase(self):
        assert agent._normalize_answer("HÀ NỘI") == "hà nội"


class TestDetectComplexity:
    def test_returns_complexity_from_llm(self, monkeypatch):
        def _fake_chat(messages, max_new_tokens=100, temperature=0.1):
            return json.dumps({"complexity": 4})

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._detect_complexity("Ai sáng lập và lãnh đạo Đảng?")
        assert result == 4

    def test_clamps_to_range(self, monkeypatch):
        def _fake_chat(messages, max_new_tokens=100, temperature=0.1):
            return json.dumps({"complexity": 10})

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._detect_complexity("test")
        assert result == 5

    def test_defaults_on_failure(self, monkeypatch):
        def _fake_chat(messages, max_new_tokens=100, temperature=0.1):
            raise RuntimeError("model error")

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._detect_complexity("test")
        assert result == 1

    def test_defaults_on_unparseable(self, monkeypatch):
        def _fake_chat(messages, max_new_tokens=100, temperature=0.1):
            return "not json"

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._detect_complexity("test")
        assert result == 1


class TestDecomposeQuestion:
    def test_returns_sub_questions(self, monkeypatch):
        def _fake_chat(messages, max_new_tokens=256, temperature=0.1):
            return json.dumps({
                "sub_questions": [
                    {"question": "Ai sáng lập Đảng?"},
                    {"question": "Đảng được thành lập năm nào?"},
                ]
            })

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._decompose_question("Ai sáng lập Đảng và năm nào?")
        assert result == ["Ai sáng lập Đảng?", "Đảng được thành lập năm nào?"]

    def test_returns_none_on_failure(self, monkeypatch):
        def _fake_chat(messages, max_new_tokens=256, temperature=0.1):
            raise RuntimeError("fail")

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._decompose_question("test")
        assert result is None

    def test_returns_none_on_invalid_format(self, monkeypatch):
        def _fake_chat(messages, max_new_tokens=256, temperature=0.1):
            return json.dumps({"sub_questions": "not a list"})

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._decompose_question("test")
        assert result is None

    def test_returns_none_on_too_many_questions(self, monkeypatch):
        def _fake_chat(messages, max_new_tokens=256, temperature=0.1):
            return json.dumps({
                "sub_questions": [{"question": f"q{i}"} for i in range(10)]
            })

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._decompose_question("test")
        assert result is None


class TestSynthesizeAnswers:
    def test_returns_synthesized_answer(self, monkeypatch):
        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            return "Hồ Chí Minh sáng lập Đảng năm 1930."

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._synthesize_answers(
            "Ai sáng lập Đảng và năm nào?",
            [("Ai sáng lập?", "Hồ Chí Minh"), ("Năm nào?", "1930")],
        )
        assert "1930" in result

    def test_returns_empty_on_failure(self, monkeypatch):
        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            raise RuntimeError("fail")

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._synthesize_answers("q", [("sq", "sa")])
        assert result == ""


class TestAgentQueryWithDecomposition:
    def test_decomposition_flow(self, monkeypatch):
        call_count = [0]

        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            call_count[0] += 1
            # Calls for sub-question agent loops:
            # Each sub-question goes through _detect_complexity (returns 1) then _agent_query_standard
            if "complexity" not in str(messages[-1].get("content", "")):
                # Check if this is a complexity detection call
                if call_count[0] in (1, 4):
                    return json.dumps({"complexity": 1})
            # Tool action
            if call_count[0] in (2, 5):
                return json.dumps({
                    "thought": "search",
                    "action": "kg_schema",
                    "action_input": {},
                })
            # Final answer for sub-questions
            if call_count[0] == 3:
                return json.dumps({"thought": "done", "final_answer": "Hồ Chí Minh"})
            if call_count[0] == 6:
                return json.dumps({"thought": "done", "final_answer": "1930"})
            # Synthesis
            return "Hồ Chí Minh sáng lập Đảng năm 1930."

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._agent_query_with_decomposition(
            "Ai sáng lập Đảng và năm nào?",
            ["Ai sáng lập Đảng?", "Đảng thành lập năm nào?"],
        )
        assert result.answer != ""


class TestRunTrajectory:
    def test_trajectory_converges(self, monkeypatch):
        call_count = [0]

        def _fake_chat(messages, max_new_tokens=512, temperature=0.7):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({
                    "thought": "search",
                    "action": "kg_schema",
                    "action_input": {},
                })
            return json.dumps({
                "thought": "found",
                "final_answer": "Trajectory answer",
            })

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._run_trajectory("test question", trajectory_id=1, temperature=0.7)
        assert result.answer == "Trajectory answer"

    def test_trajectory_with_nudge(self, monkeypatch):
        call_count = [0]

        def _fake_chat(messages, max_new_tokens=512, temperature=0.7):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({
                    "thought": "search",
                    "action": "text_search",
                    "action_input": {"query": "test"},
                })
            return json.dumps({
                "thought": "done",
                "final_answer": "Nudged answer",
            })

        class _FakeSession:
            def run(self, cypher, **params):
                return [{"page_title": "T", "page_url": "u", "chunk_id": "c1", "chunk_text": "t", "score": 1.0}]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        from contextlib import contextmanager

        @contextmanager
        def _fake_session():
            yield _FakeSession()

        monkeypatch.setattr(agent.neo4j_client, "session", _fake_session)

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._run_trajectory("test", trajectory_id=2, temperature=0.7)
        assert result.answer == "Nudged answer"

    def test_trajectory_does_not_converge(self, monkeypatch):
        def _fake_chat(messages, max_new_tokens=512, temperature=0.7):
            return "unparseable garbage"

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._run_trajectory("test", trajectory_id=0, temperature=0.7)
        assert "Không tìm thấy" in result.answer


class TestAgentQueryStandard:
    def test_sufficiency_abstain_at_iteration_5(self, monkeypatch):
        """Agent abstains when evidence is insufficient after 5 iterations."""
        call_count = [0]

        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            call_count[0] += 1
            # Always return a tool action (never final_answer) to force iterations
            return json.dumps({
                "thought": "searching",
                "action": "text_search",
                "action_input": "test query",
            })

        class _FakeSession:
            def run(self, cypher, **params):
                return []

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        from contextlib import contextmanager

        @contextmanager
        def _fake_session():
            yield _FakeSession()

        monkeypatch.setattr(agent.neo4j_client, "session", _fake_session)

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._agent_query_standard("test question")
        # Should abstain or synthesize from empty observations
        assert result.answer != ""

    def test_early_final_answer_rejected_without_tools(self, monkeypatch):
        """Agent rejects final_answer if no tools have been used yet."""
        call_count = [0]

        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            call_count[0] += 1
            if call_count[0] == 1:
                # Try to give final answer immediately
                return json.dumps({"thought": "I know", "final_answer": "Direct answer"})
            if call_count[0] == 2:
                # After rejection, use a tool
                return json.dumps({
                    "thought": "search",
                    "action": "kg_schema",
                    "action_input": {},
                })
            # Then give final answer
            return json.dumps({"thought": "done", "final_answer": "Proper answer"})

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._agent_query_standard("test")
        assert result.answer == "Proper answer"
        assert call_count[0] >= 3

    def test_unparseable_response_retried(self, monkeypatch):
        """Agent retries when LLM returns unparseable output."""
        call_count = [0]

        def _fake_chat(messages, max_new_tokens=512, temperature=0.1):
            call_count[0] += 1
            if call_count[0] == 1:
                return "This is not JSON at all"
            if call_count[0] == 2:
                return json.dumps({
                    "thought": "search",
                    "action": "kg_schema",
                    "action_input": {},
                })
            return json.dumps({"thought": "done", "final_answer": "Got it"})

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fake_chat)

        result = agent._agent_query_standard("test")
        assert result.answer == "Got it"

    def test_llm_failure_breaks_loop(self, monkeypatch):
        """Agent breaks loop when LLM fails after retry."""

        def _fail_chat(messages, max_new_tokens=512, temperature=0.1):
            raise RuntimeError("model unavailable")

        import src.infrastructure.local_llm
        monkeypatch.setattr(src.infrastructure.local_llm, "chat", _fail_chat)

        result = agent._agent_query_standard("test")
        # Should return synthesized/fallback answer
        assert result.answer != ""
