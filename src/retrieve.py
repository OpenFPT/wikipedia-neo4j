"""Graph retrieval and deterministic answer assembly."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.llm import assert_readonly_cypher, embed_texts, generate_readonly_cypher
from src.logging_utils import get_logger
from src.neo4j_client import neo4j_client


logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Query response model used by API layer."""

    answer: str
    citations: list[dict]


_LEGACY_HYBRID_CYPHER = """
CALL {
  CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node, score
  MATCH (p:Page)-[:HAS_CHUNK]->(node)
  RETURN p.title AS page_title,
         p.url AS page_url,
         p.id AS page_id,
         node.id AS chunk_id,
         node.text AS chunk_text,
         score * 1.0 AS score
  LIMIT $top_k

  UNION

  CALL db.index.fulltext.queryNodes('page_title_ft', $q) YIELD node, score
  MATCH (node:Page)-[:HAS_CHUNK]->(c:Chunk)
  RETURN node.title AS page_title,
         node.url AS page_url,
         node.id AS page_id,
         c.id AS chunk_id,
         c.text AS chunk_text,
         score * 0.8 AS score
  LIMIT $top_k

  UNION

  CALL db.index.fulltext.queryNodes('entity_alias_ft', $q) YIELD node, score
  MATCH (c:Chunk)-[:MENTIONS]->(node)
  MATCH (p:Page)-[:HAS_CHUNK]->(c)
  RETURN p.title AS page_title,
         p.url AS page_url,
         p.id AS page_id,
         c.id AS chunk_id,
         c.text AS chunk_text,
         score * 0.7 AS score
  LIMIT $top_k
}
WITH page_title, page_url, page_id, chunk_id, chunk_text, max(score) AS score
RETURN page_title, page_url, page_id, chunk_id, chunk_text, score
ORDER BY score DESC
LIMIT $top_k
"""

_BM25_CYPHER = """
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node, score
MATCH (p:Page)-[:HAS_CHUNK]->(node)
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       node.id AS chunk_id,
       node.text AS chunk_text,
       score AS bm25_score
ORDER BY score DESC
LIMIT $top_k
"""

_VECTOR_CYPHER = """
CALL db.index.vector.queryNodes('chunk_embedding_idx', $top_k, $embedding)
YIELD node AS c, score AS similarity
MATCH (p:Page)-[:HAS_CHUNK]->(c)
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       similarity AS vector_score
"""

_VECTOR_SEARCH_CYPHER25 = """
SEARCH chunk_embedding_idx
FOR VECTOR NEAREST (c:Chunk) TO $query_embedding
YIELD c, score
MATCH (p:Page)-[:HAS_CHUNK]->(c)
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS vector_score
ORDER BY score DESC
LIMIT $top_k
"""

_GRAPH_CYPHER = """
CALL db.index.fulltext.queryNodes('entity_alias_ft', $q) YIELD node AS e, score
MATCH (c:Chunk)-[:MENTIONS]->(e)
MATCH (p:Page)-[:HAS_CHUNK]->(c)
RETURN DISTINCT p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS graph_score
ORDER BY score DESC
LIMIT $top_k
"""

_EXPAND_LINKS_CYPHER = """
MATCH (source:Page)-[:LINKS_TO]->(linked:Page)-[:HAS_CHUNK]->(c:Chunk)
WHERE source.id IN $page_ids
WITH linked, c
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node, score
WHERE node = c
MATCH (linked)-[:HAS_CHUNK]->(node)
RETURN linked.title AS page_title,
       linked.url AS page_url,
       linked.id AS page_id,
       node.id AS chunk_id,
       node.text AS chunk_text,
       score * 0.9 AS score
ORDER BY score DESC
LIMIT $top_k
"""


def _run_bm25_query(question: str, top_k: int) -> list[dict]:  # pragma: no cover
    """Execute BM25 fulltext search."""
    with neo4j_client.session() as session:
        records = session.run(_BM25_CYPHER, q=question, top_k=top_k)
        rows = [dict(r) for r in records]
    logger.info("BM25 retrieval executed", extra={"rows": len(rows)})
    return rows


def _run_vector_query(
    question: str, top_k: int, *, query_embedding: list[float] | None = None
) -> list[dict]:  # pragma: no cover
    """Execute vector similarity search."""
    if settings.neo4j_use_search_clause:
        return _run_vector_query_cypher25(question, top_k, query_embedding=query_embedding)
    try:
        embedding = query_embedding or embed_texts([question])[0]

        with neo4j_client.session() as session:
            records = session.run(_VECTOR_CYPHER, embedding=embedding, top_k=top_k)
            rows = [dict(r) for r in records]
        logger.info("Vector retrieval executed", extra={"rows": len(rows)})
        return rows
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
        return []


def _run_vector_query_cypher25(
    question: str, top_k: int, *, query_embedding: list[float] | None = None
) -> list[dict]:  # pragma: no cover
    """Vector search using Neo4j Cypher 25 SEARCH clause (requires Neo4j 2026.02+)."""
    try:
        embedding = query_embedding or embed_texts([question])[0]

        with neo4j_client.session() as session:
            records = session.run(
                _VECTOR_SEARCH_CYPHER25,
                query_embedding=embedding,
                top_k=top_k,
            )
            rows = [dict(r) for r in records]
        logger.info("Vector retrieval (Cypher 25 SEARCH) executed", extra={"rows": len(rows)})
        return rows
    except Exception as e:
        logger.warning(f"Cypher 25 SEARCH vector query failed: {e}")
        return []


def _run_graph_query(question: str, top_k: int) -> list[dict]:  # pragma: no cover
    """Execute graph-based entity search."""
    try:
        with neo4j_client.session() as session:
            records = session.run(_GRAPH_CYPHER, q=question, top_k=top_k)
            rows = [dict(r) for r in records]
        logger.info("Graph retrieval executed", extra={"rows": len(rows)})
        return rows
    except Exception as e:
        logger.warning(f"Graph search failed: {e}")
        return []


def _community_search(
    question: str, top_k: int, *, query_embedding: list[float] | None = None
) -> list[dict]:  # pragma: no cover
    """Search community summaries for broad/global questions."""
    try:
        embedding = query_embedding or embed_texts([question])[0]
    except Exception as e:
        logger.warning(f"Community search embedding failed: {e}")
        return []

    try:
        with neo4j_client.session() as session:
            records = session.run(
                """
                MATCH (cm:Community)
                WHERE cm.embedding IS NOT NULL
                WITH cm, vector.similarity.cosine(cm.embedding, $query_embedding) AS score
                ORDER BY score DESC
                LIMIT $top_k
                MATCH (cm)-[:HAS_MEMBER]->(e:Entity)<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(p:Page)
                WITH p, c, score, cm
                RETURN DISTINCT p.title AS page_title, p.url AS page_url, p.id AS page_id,
                       c.id AS chunk_id, c.text AS chunk_text, score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                query_embedding=embedding,
                top_k=top_k,
            )
            rows = [dict(r) for r in records]
        logger.info("Community search executed", extra={"rows": len(rows)})
        return rows
    except Exception as e:
        logger.warning(f"Community search failed: {e}")
        return []


def _wrrf_fusion(
    results_list: list[list[dict]],
    weights: list[float],
    k: int = 60,
) -> list[dict]:
    """Weighted Reciprocal Rank Fusion.

    Formula: score(d) = sum(weight_i / (k + rank_i(d))) for each signal i
    Deduplicates by chunk_id, keeps highest fused score.
    """
    # Build rank maps for each signal (1-based ranks)
    rank_maps: list[dict[str, int]] = []
    for results in results_list:
        ranks = {r["chunk_id"]: i + 1 for i, r in enumerate(results)}
        rank_maps.append(ranks)

    # Collect all chunk IDs across all signals
    all_chunk_ids: set[str] = set()
    for rm in rank_maps:
        all_chunk_ids.update(rm.keys())

    # Compute WRRF scores
    wrrf_scores: dict[str, float] = {}
    for chunk_id in all_chunk_ids:
        score = 0.0
        for weight, rm in zip(weights, rank_maps):
            if chunk_id in rm:
                score += weight / (k + rm[chunk_id])
        wrrf_scores[chunk_id] = score

    # Build result metadata map (first occurrence wins)
    result_map: dict[str, dict] = {}
    for results in results_list:
        for r in results:
            chunk_id = r["chunk_id"]
            if chunk_id not in result_map:
                result_map[chunk_id] = {
                    "page_title": r.get("page_title", ""),
                    "page_url": r.get("page_url", ""),
                    "page_id": r.get("page_id", ""),
                    "chunk_id": chunk_id,
                    "chunk_text": r.get("chunk_text", ""),
                }

    # Sort by WRRF score descending
    sorted_results = sorted(
        [{**result_map[cid], "score": wrrf_scores[cid]} for cid in wrrf_scores],
        key=lambda x: x["score"],
        reverse=True,
    )

    logger.info(
        "WRRF fusion completed",
        extra={
            "signals": len(results_list),
            "candidates": len(all_chunk_ids),
            "output": len(sorted_results),
        },
    )

    return sorted_results


def _run_fallback_query(question: str, top_k: int) -> list[dict]:
    """Execute WRRF hybrid search as fallback when LLM-generated Cypher fails.

    Combines BM25, vector, graph, and community signals via Weighted Reciprocal Rank Fusion.
    Falls back to legacy hybrid Cypher if all signals fail.
    """
    # Embed question once for vector and community signals
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
    all_weights = [settings.wrrf_weight_bm25, settings.wrrf_weight_vector, settings.wrrf_weight_graph]

    if community_results:
        all_results.append(community_results)
        all_weights.append(settings.wrrf_weight_community)

    if not any(all_results):
        # All signals failed; fall back to legacy single-query hybrid
        logger.warning("WRRF signals all empty, using legacy hybrid fallback")
        return _run_legacy_fallback_query(question, top_k)

    fused = _wrrf_fusion(
        all_results,
        all_weights,
        k=settings.wrrf_k,
    )
    return fused[:top_k]


def _run_legacy_fallback_query(question: str, top_k: int) -> list[dict]:
    """Execute legacy hybrid UNION query as last-resort fallback."""
    with neo4j_client.session() as session:
        records = session.run(
            _LEGACY_HYBRID_CYPHER,
            q=question,
            top_k=top_k,
        )
        rows = [dict(r) for r in records]
    logger.info("Legacy fallback retrieval executed", extra={"rows": len(rows)})
    return rows


def _expand_via_links(page_ids: list[str], question: str, top_k: int) -> list[dict]:
    """Retrieve chunks from pages connected via LINKS_TO edges."""
    if not page_ids:
        return []
    with neo4j_client.session() as session:
        records = session.run(
            _EXPAND_LINKS_CYPHER,
            page_ids=page_ids,
            q=question,
            top_k=top_k,
        )
        rows = [dict(r) for r in records]
    logger.info("Multi-hop expansion", extra={"source_pages": len(page_ids), "expanded_rows": len(rows)})
    return rows


def _run_generated_query(question: str, top_k: int) -> list[dict]:
    """Generate, validate, and execute LLM-produced read-only Cypher."""
    cypher = generate_readonly_cypher(question)
    assert_readonly_cypher(cypher)

    with neo4j_client.session() as session:
        records = session.run(cypher, q=question, top_k=top_k)
        rows = [dict(r) for r in records]

    required_keys = {"page_title", "page_url", "chunk_id", "chunk_text", "score"}
    for row in rows:
        if not required_keys.issubset(row):
            raise RuntimeError("Generated query returned unexpected shape")

    logger.info("Generated retrieval executed", extra={"rows": len(rows)})
    return rows


def query_graph(question: str, top_k: int = 4) -> QueryResult:
    """Query graph and synthesize a deterministic answer with citations."""
    if settings.model_mode == "local":
        from src.agent import agent_query

        return agent_query(question, top_k)

    try:
        rows = _run_generated_query(question, top_k)
    except (RuntimeError, ValueError, KeyError, TypeError):
        # Fall back to hybrid search
        rows = _run_fallback_query(question, top_k)

    if not rows:
        return QueryResult(
            answer="I could not find relevant context in the graph yet. Try ingesting more topics.",
            citations=[],
        )

    from src.reranker import rerank

    reranked = rerank(question, rows, text_key="chunk_text", top_k=top_k)

    if settings.multi_hop_expansion and reranked:
        page_ids = list({r["page_id"] for r in reranked if r.get("page_id")})
        expanded = _expand_via_links(page_ids, question, top_k)
        if expanded:
            seen_chunks = {r["chunk_id"] for r in reranked}
            new_rows = [r for r in expanded if r["chunk_id"] not in seen_chunks]
            if new_rows:
                combined = reranked + new_rows
                reranked = rerank(question, combined, text_key="chunk_text", top_k=top_k)

    citations = [
        {
            "page_title": r["page_title"],
            "page_url": r["page_url"],
            "chunk_id": r["chunk_id"],
        }
        for r in reranked
    ]

    snippets = []
    for r in reranked:
        txt = (r["chunk_text"] or "").strip().replace("\n", " ")
        snippets.append(txt[:220])

    answer = "Deterministic demo answer from retrieved graph context: " + " | ".join(snippets[:2])

    return QueryResult(answer=answer, citations=citations)
