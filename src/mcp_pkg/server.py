"""Standalone MCP server entry point for Claude Desktop/Code integration."""

from __future__ import annotations

import argparse

from fastmcp import FastMCP

from src.mcp_pkg.tools import register_tools


def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server instance."""
    mcp = FastMCP(
        "WikiGraphRAG",
        instructions=(
            "Vietnamese Wikipedia knowledge graph QA system. "
            "Use search_knowledge_base for simple factual lookups. "
            "Use answer_question for complex multi-hop questions. "
            "Use explore_entity and find_path for graph exploration."
        ),
    )
    register_tools(mcp)
    return mcp


mcp = create_mcp_server()


def main() -> None:
    parser = argparse.ArgumentParser(description="WikiGraphRAG MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport mode (default: stdio for Claude Desktop/Code)",
    )
    parser.add_argument("--port", type=int, default=8001, help="Port for HTTP transport")
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transport")
    args = parser.parse_args()

    if args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
