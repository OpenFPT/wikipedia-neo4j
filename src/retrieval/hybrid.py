"""Graph retrieval and deterministic answer assembly.

This module re-exports from the split submodules for backward compatibility.
"""

from src.retrieval.bm25 import run_bm25_query as _run_bm25_query  # noqa: F401
from src.retrieval.community import community_search as _community_search  # noqa: F401
from src.retrieval.fusion import (  # noqa: F401
    QueryResult,
    _run_fallback_query,
    _run_generated_query,
    _run_legacy_fallback_query,
    _synthesize_answer,
    _wrrf_fuse,
    _wrrf_fusion,
    hybrid_retrieve,
    query_graph,
)
from src.retrieval.graph import (  # noqa: F401
    expand_via_links as _expand_via_links,
    graph_search as _graph_search,
    run_graph_query as _run_graph_query,
)
from src.retrieval.vector import (  # noqa: F401
    run_vector_query as _run_vector_query,
    run_vector_query_cypher25 as _run_vector_query_cypher25,
    vector_search as _vector_search,
)
