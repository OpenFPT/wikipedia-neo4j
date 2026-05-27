"""Tests for MCP server tool registration and basic functionality."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from fastmcp import FastMCP

from src.mcp_tools import register_tools


def _create_test_mcp() -> FastMCP:
    mcp = FastMCP("TestWikiGraphRAG")
    register_tools(mcp)
    return mcp


def _list_tool_names(mcp: FastMCP) -> list[str]:
    tools = asyncio.run(mcp.list_tools())
    return [t.name for t in tools]


def _get_tool_fn(mcp: FastMCP, name: str):
    tool = asyncio.run(mcp.get_tool(name))
    return tool.fn


def test_all_tools_registered():
    mcp = _create_test_mcp()
    tool_names = _list_tool_names(mcp)
    assert "search_knowledge_base" in tool_names
    assert "explore_entity" in tool_names
    assert "find_path" in tool_names
    assert "get_community_summary" in tool_names
    assert "answer_question" in tool_names
    assert "get_graph_stats" in tool_names
    assert "list_entity_types" in tool_names
    assert "kg_query" in tool_names
    assert len(tool_names) == 9


def test_graph_schema_resource_registered():
    mcp = _create_test_mcp()
    resources = asyncio.run(mcp.list_resources())
    uris = [str(r.uri) for r in resources]
    assert any("schema" in u for u in uris)


@patch("src.mcp_tools.hybrid_retrieve")
def test_search_knowledge_base_returns_results(mock_retrieve):
    mock_retrieve.return_value = [
        {
            "chunk_id": "c1",
            "chunk_text": "Hồ Chí Minh sinh năm 1890",
            "score": 0.95,
            "page_title": "Hồ Chí Minh",
            "page_url": "https://vi.wikipedia.org/wiki/Hồ_Chí_Minh",
        }
    ]
    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "search_knowledge_base")
    result = tool_fn(question="Hồ Chí Minh sinh năm nào?", top_k=3, method="hybrid")
    assert result["total"] == 1
    assert result["results"][0]["page_title"] == "Hồ Chí Minh"
    mock_retrieve.assert_called_once_with("Hồ Chí Minh sinh năm nào?", top_k=3)


@patch("src.mcp_tools.neo4j_client")
def test_get_graph_stats_returns_counts(mock_neo4j):
    mock_session = MagicMock()
    mock_record = {"pages": 100, "chunks": 500, "entities": 200}
    mock_session.run.return_value.single.return_value = mock_record
    mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "get_graph_stats")
    result = tool_fn()
    assert result["pages"] == 100
    assert result["chunks"] == 500
    assert result["entities"] == 200


@patch("src.mcp_tools.neo4j_client")
def test_explore_entity_handles_not_found(mock_neo4j):
    mock_session = MagicMock()
    mock_session.run.return_value = iter([])
    mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "explore_entity")
    result = tool_fn(entity_name="NonExistent", depth=1)
    assert result["count"] == 0
    assert result["neighbors"] == []


@patch("src.mcp_tools.hybrid_retrieve")
def test_search_handles_error_gracefully(mock_retrieve):
    mock_retrieve.side_effect = RuntimeError("Neo4j connection failed")

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "search_knowledge_base")
    result = tool_fn(question="test", top_k=5, method="hybrid")
    assert "error" in result
    assert result["total"] == 0


@patch("src.mcp_tools.neo4j_client")
def test_find_path_returns_path(mock_neo4j):
    mock_session = MagicMock()
    mock_record = {"nodes": ["A", "B", "C"], "relations": ["KNOWS", "LOCATED_IN"]}
    mock_session.run.return_value.single.return_value = mock_record
    mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "find_path")
    result = tool_fn(entity_a="A", entity_b="C", max_hops=5)
    assert result["path"] == ["A", "B", "C"]
    assert result["hops"] == 2


@patch("src.mcp_tools.neo4j_client")
def test_find_path_no_path(mock_neo4j):
    mock_session = MagicMock()
    mock_session.run.return_value.single.return_value = None
    mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "find_path")
    result = tool_fn(entity_a="X", entity_b="Y", max_hops=3)
    assert result["path"] is None
    assert "No path found" in result["message"]


@patch("src.mcp_tools.neo4j_client")
def test_find_path_handles_error(mock_neo4j):
    mock_neo4j.session.return_value.__enter__ = MagicMock(
        side_effect=RuntimeError("connection lost")
    )
    mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "find_path")
    result = tool_fn(entity_a="A", entity_b="B", max_hops=5)
    assert "error" in result


@patch("src.mcp_tools.neo4j_client")
def test_list_entity_types(mock_neo4j):
    mock_session = MagicMock()
    mock_session.run.return_value = iter([
        {"type": "Person", "count": 50},
        {"type": "Location", "count": 30},
    ])
    mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "list_entity_types")
    result = tool_fn()
    assert result["total_types"] == 2
    assert result["entity_types"][0]["type"] == "Person"


@patch("src.mcp_tools.neo4j_client")
def test_list_entity_types_handles_error(mock_neo4j):
    mock_neo4j.session.return_value.__enter__ = MagicMock(
        side_effect=RuntimeError("db error")
    )
    mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "list_entity_types")
    result = tool_fn()
    assert "error" in result
    assert result["total_types"] == 0


@patch("src.mcp_tools.neo4j_client")
def test_source_trace_returns_context(mock_neo4j):
    mock_session = MagicMock()
    mock_record = {
        "page_title": "Test Page",
        "page_url": "https://vi.wikipedia.org/wiki/Test",
        "chunk_text": "Some text content",
        "seq": 2,
        "neighbors": [
            {"id": "c1", "text": "prev chunk", "seq": 1},
            {"id": "c2", "text": "Some text content", "seq": 2},
            {"id": "c3", "text": "next chunk", "seq": 3},
        ],
    }
    mock_session.run.return_value.single.return_value = mock_record
    mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "source_trace")
    result = tool_fn(chunk_id="c2")
    assert result["page_title"] == "Test Page"
    assert result["sequence_number"] == 2
    assert len(result["neighbors"]) == 3


@patch("src.mcp_tools.neo4j_client")
def test_source_trace_not_found(mock_neo4j):
    mock_session = MagicMock()
    mock_session.run.return_value.single.return_value = None
    mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "source_trace")
    result = tool_fn(chunk_id="nonexistent")
    assert "Chunk not found" in result.get("error", "")


@patch("src.mcp_tools.neo4j_client")
def test_source_trace_handles_error(mock_neo4j):
    mock_neo4j.session.return_value.__enter__ = MagicMock(
        side_effect=RuntimeError("timeout")
    )
    mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "source_trace")
    result = tool_fn(chunk_id="c1")
    assert "error" in result


@patch("src.mcp_tools.neo4j_client")
def test_kg_query_valid_read(mock_neo4j):
    mock_session = MagicMock()
    mock_session.run.return_value = iter([
        {"name": "Hà Nội", "type": "Location"},
    ])
    mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "kg_query")
    result = tool_fn(cypher="MATCH (e:Entity) RETURN e.name AS name, e.type AS type LIMIT 1", params=None)
    assert result["total"] == 1
    assert result["results"][0]["name"] == "Hà Nội"


def test_kg_query_rejects_write():
    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "kg_query")
    result = tool_fn(cypher="CREATE (n:Entity {name: 'bad'})", params=None)
    assert "error" in result
    assert "Write operation" in result["error"]


def test_kg_query_rejects_empty():
    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "kg_query")
    result = tool_fn(cypher="", params=None)
    assert "error" in result
    assert "empty" in result["error"]


def test_kg_query_rejects_multi_statement():
    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "kg_query")
    result = tool_fn(cypher="MATCH (n) RETURN n; MATCH (m) RETURN m", params=None)
    assert "error" in result
    assert "Multiple statements" in result["error"]


@patch("src.mcp_tools.neo4j_client")
def test_kg_query_handles_db_error(mock_neo4j):
    mock_session = MagicMock()
    mock_session.run.side_effect = RuntimeError("syntax error")
    mock_neo4j.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_neo4j.session.return_value.__exit__ = MagicMock(return_value=False)

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "kg_query")
    result = tool_fn(cypher="MATCH (n) RETURN n LIMIT 1", params=None)
    assert "error" in result


@patch("src.community.get_community_for_entity", return_value=42)
@patch("src.community.get_community_summary", return_value="Summary about topic")
def test_get_community_summary_found(mock_summ, mock_comm):
    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "get_community_summary")
    result = tool_fn(topic="Chiến tranh Việt Nam")
    assert result["community_id"] == 42
    assert result["summary"] == "Summary about topic"


@patch("src.community.get_community_for_entity", return_value=None)
def test_get_community_summary_not_found(mock_comm):
    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "get_community_summary")
    result = tool_fn(topic="NonExistentTopic")
    assert result["summary"] is None
    assert "No community found" in result["message"]


@patch("src.community.get_community_for_entity", side_effect=RuntimeError("fail"))
def test_get_community_summary_handles_error(mock_comm):
    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "get_community_summary")
    result = tool_fn(topic="test")
    assert "error" in result


@patch("src.agent.run_agent_scaled")
def test_answer_question_success(mock_agent):
    mock_result = MagicMock()
    mock_result.answer = "Hồ Chí Minh sinh năm 1890"
    mock_result.citations = [{"page": "HCM"}]
    mock_result.retrieval_tier = "simple"
    mock_agent.return_value = mock_result

    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "answer_question")
    result = tool_fn(question="Hồ Chí Minh sinh năm nào?")
    assert result["answer"] == "Hồ Chí Minh sinh năm 1890"
    assert result["retrieval_tier"] == "simple"


@patch("src.agent.run_agent_scaled", side_effect=RuntimeError("model unavailable"))
def test_answer_question_handles_error(mock_agent):
    mcp = _create_test_mcp()
    tool_fn = _get_tool_fn(mcp, "answer_question")
    result = tool_fn(question="test")
    assert "error" in result
    assert result["answer"] == ""


def test_validate_readonly_cypher_allows_read():
    from src.mcp_tools import _validate_readonly_cypher

    _validate_readonly_cypher("MATCH (n) RETURN n LIMIT 10")
    _validate_readonly_cypher("MATCH (a)-[r]->(b) WHERE a.name = 'test' RETURN a, r, b")


def test_validate_readonly_cypher_blocks_writes():
    from src.mcp_tools import _validate_readonly_cypher
    import pytest

    with pytest.raises(ValueError, match="Write operation"):
        _validate_readonly_cypher("CREATE (n:Node {name: 'x'})")
    with pytest.raises(ValueError, match="Write operation"):
        _validate_readonly_cypher("MATCH (n) DELETE n")
    with pytest.raises(ValueError, match="Write operation"):
        _validate_readonly_cypher("MATCH (n) DETACH DELETE n")
    with pytest.raises(ValueError, match="Write operation"):
        _validate_readonly_cypher("MERGE (n:Node {id: 1})")
    with pytest.raises(ValueError, match="Write operation"):
        _validate_readonly_cypher("MATCH (n) SET n.x = 1")
    with pytest.raises(ValueError, match="Write operation"):
        _validate_readonly_cypher("MATCH (n) REMOVE n.x")


def test_validate_readonly_cypher_blocks_empty():
    from src.mcp_tools import _validate_readonly_cypher
    import pytest

    with pytest.raises(ValueError, match="empty"):
        _validate_readonly_cypher("")
    with pytest.raises(ValueError, match="empty"):
        _validate_readonly_cypher("   ")


def test_validate_readonly_cypher_blocks_multi_statement():
    from src.mcp_tools import _validate_readonly_cypher
    import pytest

    with pytest.raises(ValueError, match="Multiple statements"):
        _validate_readonly_cypher("MATCH (n) RETURN n; DROP INDEX idx")


def test_mcp_server_module_importable():
    from src.mcp_server import create_mcp_server, mcp

    assert mcp is not None
    server = create_mcp_server()
    tools = asyncio.run(server.list_tools())
    assert len(tools) == 9
