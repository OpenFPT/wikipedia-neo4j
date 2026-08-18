"""Backward-compatible retrieval API shim."""

from __future__ import annotations

from neo4j.exceptions import CypherSyntaxError

from src.config import settings
from src.infrastructure.llm import (
    _client_pool,
    assert_readonly_cypher,
    embed_texts,
    generate_readonly_cypher,
)
from src.infrastructure.neo4j_client import neo4j_client
from src.retrieval.bm25 import run_bm25_query as _run_bm25_query
from src.retrieval.community import community_search as _community_search
from src.retrieval.fusion import (
    QueryResult,
    QueryTrace,
    QueryTraceStep,
    _LEGACY_HYBRID_CYPHER,
    _wrrf_fuse,
    _wrrf_fusion,
    hybrid_retrieve,
)
from src.retrieval.graph import (
    expand_via_links as _expand_via_links,
    graph_search as _graph_search,
    run_graph_query as _run_graph_query,
)
import src.retrieval.reranker as _reranker_mod
from src.retrieval.vector import (
    run_vector_query as _run_vector_query,
    run_vector_query_cypher25 as _run_vector_query_cypher25,
    vector_search as _vector_search,
)

_DEFAULT_NEO4J_CLIENT = neo4j_client
_DEFAULT_RUN_BM25_QUERY = _run_bm25_query
_DEFAULT_RUN_VECTOR_QUERY = _run_vector_query
_DEFAULT_RUN_GRAPH_QUERY = _run_graph_query
_DEFAULT_COMMUNITY_SEARCH = _community_search


def _synthesize_answer(question: str, snippets: list[str]) -> str:
    """Deterministic answer assembly kept for compatibility with older tests."""
    if not snippets:
        return "Không đủ thông tin trong dữ liệu hiện có để trả lời chính xác."

    preview_parts = []
    for snippet in snippets[:3]:
        text = (snippet or "").strip()
        if len(text) > 180:
            text = text[:177].rstrip() + "..."
        if text:
            preview_parts.append(text)

    if not preview_parts:
        return "Không đủ thông tin trong dữ liệu hiện có để trả lời chính xác."

    return "Dựa trên thông tin tìm được: " + " ".join(preview_parts)


def _run_fallback_query(question: str, top_k: int) -> list[dict]:
    """Execute the legacy shim fallback using monkeypatchable module symbols."""
    uses_default_signal_stack = (
        _run_bm25_query is _DEFAULT_RUN_BM25_QUERY
        and _run_vector_query is _DEFAULT_RUN_VECTOR_QUERY
        and _run_graph_query is _DEFAULT_RUN_GRAPH_QUERY
        and _community_search is _DEFAULT_COMMUNITY_SEARCH
    )
    if neo4j_client is not _DEFAULT_NEO4J_CLIENT and uses_default_signal_stack:
        return _run_legacy_fallback_query(question, top_k)

    try:
        query_embedding = embed_texts([question])[0]
    except Exception:
        query_embedding = None

    bm25_results = _run_bm25_query(question, top_k * 3)
    vector_results = (
        _run_vector_query(question, top_k * 3, query_embedding=query_embedding)
        if query_embedding
        else []
    )
    graph_results = _run_graph_query(question, top_k * 3)
    community_results = (
        _community_search(question, top_k * 2, query_embedding=query_embedding)
        if query_embedding
        else []
    )

    all_results = [bm25_results, vector_results, graph_results]
    all_weights = [
        settings.wrrf_weight_bm25,
        settings.wrrf_weight_vector,
        settings.wrrf_weight_graph,
    ]

    if community_results:
        all_results.append(community_results)
        all_weights.append(settings.wrrf_weight_community)

    if not any(all_results):
        return _run_legacy_fallback_query(question, top_k)

    return _wrrf_fusion(all_results, all_weights, k=settings.wrrf_k)[:top_k]


def _run_legacy_fallback_query(question: str, top_k: int) -> list[dict]:
    with neo4j_client.session() as session:
        records = session.run(_LEGACY_HYBRID_CYPHER, q=question, top_k=top_k)
        return [dict(r) for r in records]


def _run_generated_query(question: str, top_k: int, emit=None) -> list[dict]:
    cypher = generate_readonly_cypher(question)
    if emit is not None:
        emit("tool_call", {"tool": "generated_query", "input": {"question": question}})
        emit("cypher", {"query": cypher})
    assert_readonly_cypher(cypher)

    with neo4j_client.session() as session:
        records = session.run(cypher, q=question, top_k=top_k)
        rows = [dict(r) for r in records]

    required_keys = {"page_title", "page_url", "chunk_id", "chunk_text", "score"}
    for row in rows:
        if not required_keys.issubset(row):
            raise RuntimeError("Generated query returned unexpected shape")

    if emit is not None:
        emit("tool_result", {"tool": "generated_query", "row_count": len(rows), "status": "ok"})
    return rows


def query_graph(question: str, top_k: int = 4, emit=None) -> QueryResult:
    retrieval_tier = "generated"
    trace_steps: list[QueryTraceStep] = []

    if emit is not None:
        emit("route", {"route": "generated"})

    try:
        rows = _run_generated_query(question, top_k, emit=emit)
        trace_steps.append(
            {
                "kind": "retrieval",
                "name": "generated_query",
                "status": "ok",
                "row_count": len(rows),
                "top_k": top_k,
            }
        )
    except (RuntimeError, ValueError, KeyError, TypeError, CypherSyntaxError) as exc:
        retrieval_tier = "wrrf"
        if emit is not None:
            emit(
                "fallback",
                {"name": "wrrf_fallback", "error_type": type(exc).__name__, "message": str(exc)},
            )
        trace_steps.append(
            {
                "kind": "retrieval",
                "name": "generated_query_failed",
                "status": "fallback",
                "error_type": type(exc).__name__,
            }
        )
        rows = _run_fallback_query(question, top_k)
        if emit is not None:
            emit("tool_result", {"tool": "wrrf_fallback", "row_count": len(rows), "status": "ok"})
        trace_steps.append(
            {
                "kind": "retrieval",
                "name": "wrrf_fallback",
                "status": "ok",
                "row_count": len(rows),
                "top_k": top_k,
            }
        )

    trace: QueryTrace = {"tier": retrieval_tier, "steps": trace_steps}

    if not rows:
        return QueryResult(
            answer="I could not find relevant context in the graph yet. Try ingesting more topics.",
            citations=[],
            retrieval_tier=retrieval_tier,
            trace=trace,
        )

    reranked = _reranker_mod.rerank(question, rows, text_key="chunk_text", top_k=top_k)

    if settings.multi_hop_expansion and reranked:
        page_ids = list({r["page_id"] for r in reranked if r.get("page_id")})
        expanded = _expand_via_links(page_ids, question, top_k)
        if expanded:
            seen_chunks = {r["chunk_id"] for r in reranked}
            new_rows = [r for r in expanded if r["chunk_id"] not in seen_chunks]
            if new_rows:
                reranked = _reranker_mod.rerank(question, reranked + new_rows, text_key="chunk_text", top_k=top_k)

    citations = [
        {
            "page_title": r["page_title"],
            "page_url": r["page_url"],
            "chunk_id": r["chunk_id"],
        }
        for r in reranked
    ]
    answer = _synthesize_answer(question, [r.get("chunk_text", "") for r in reranked])
    return QueryResult(
        answer=answer,
        citations=citations,
        retrieval_tier=retrieval_tier,
        trace=trace,
    )
