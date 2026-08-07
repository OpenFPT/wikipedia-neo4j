"""Graph retrieval and deterministic answer assembly.

This module re-exports from the split submodules for backward compatibility.
"""

# Re-export public interface from submodules
from src.retrieval.fusion import (  # noqa: F401
    QueryResult,
    hybrid_retrieve,
    query_graph,
    _wrrf_fusion,
    _wrrf_fuse,
    _run_fallback_query,
    _run_generated_query,
    _run_legacy_fallback_query,
    _synthesize_answer,
)
from src.retrieval.bm25 import run_bm25_query as _run_bm25_query  # noqa: F401
from src.retrieval.vector import (  # noqa: F401
    run_vector_query as _run_vector_query,
    run_vector_query_cypher25 as _run_vector_query_cypher25,
    vector_search as _vector_search,
)
from src.retrieval.graph import (  # noqa: F401
    run_graph_query as _run_graph_query,
    graph_search as _graph_search,
    expand_via_links as _expand_via_links,
)
from src.retrieval.community import community_search as _community_search  # noqa: F401
