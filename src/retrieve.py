"""Backward-compatibility shim: imports redirect to src.retrieval submodules."""

from src.config import settings  # noqa: F401
from src.infrastructure.neo4j_client import neo4j_client  # noqa: F401

from src.retrieval.fusion import (  # noqa: F401
    QueryResult,
    hybrid_retrieve,
    query_graph,
    _wrrf_fusion,
    _wrrf_fuse,
    _run_fallback_query,
    _run_generated_query,
)
from src.retrieval.vector import vector_search as _vector_search  # noqa: F401
from src.retrieval.graph import (  # noqa: F401
    graph_search as _graph_search,
    expand_via_links as _expand_via_links,
)
from src.retrieval.bm25 import run_bm25_query as _run_bm25_query  # noqa: F401
from src.retrieval.community import community_search as _community_search  # noqa: F401
