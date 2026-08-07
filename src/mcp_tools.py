"""Backward-compatibility shim: imports redirect to src.mcp_pkg.tools."""

from src.mcp_pkg.tools import (  # noqa: F401
    register_tools,
    _validate_readonly_cypher,
)
