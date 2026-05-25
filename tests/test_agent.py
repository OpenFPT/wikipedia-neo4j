"""Tests for ReAct agent loop and tools."""

from __future__ import annotations

import json

import src.agent as agent


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

        import src.local_llm
        monkeypatch.setattr(src.local_llm, "chat", _fake_chat)

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

        import src.local_llm
        monkeypatch.setattr(src.local_llm, "chat", _fake_chat)

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

        import src.local_llm
        monkeypatch.setattr(src.local_llm, "chat", _fake_chat)

        result = agent.agent_query("test")
        assert result.answer == "fallback answer"
