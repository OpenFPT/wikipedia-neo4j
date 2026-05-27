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


def test_mcp_server_module_importable():
    from src.mcp_server import create_mcp_server, mcp

    assert mcp is not None
    server = create_mcp_server()
    tools = asyncio.run(server.list_tools())
    assert len(tools) == 9
