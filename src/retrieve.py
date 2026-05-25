"""Graph retrieval and deterministic answer assembly."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.llm import assert_readonly_cypher, generate_readonly_cypher
from src.logging_utils import get_logger
from src.neo4j_client import neo4j_client


logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Query response model used by API layer."""

    answer: str
    citations: list[dict]


_HYBRID_FALLBACK_CYPHER = """
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
MATCH (c:Chunk)
WITH c, vector.similarity.cosine(c.embedding, $embedding) AS similarity
WHERE similarity > 0.3
MATCH (p:Page)-[:HAS_CHUNK]->(c)
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       similarity AS vector_score
ORDER BY similarity DESC
LIMIT $top_k
"""

_GRAPH_CYPHER = """
MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE e.alias CONTAINS $q
MATCH (p2:Page)-[:HAS_CHUNK]->(c2:Chunk)-[:MENTIONS]->(e)
RETURN DISTINCT p2.title AS page_title,
       p2.url AS page_url,
       p2.id AS page_id,
       c2.id AS chunk_id,
       c2.text AS chunk_text,
       1.0 AS graph_score
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


def _run_vector_query(question: str, top_k: int) -> list[dict]:  # pragma: no cover
    """Execute vector similarity search."""
    try:
        from src.llm import embed_texts

        embeddings = embed_texts([question])
        embedding = embeddings[0] if embeddings else None
        if not embedding:
            return []

        with neo4j_client.session() as session:
            records = session.run(_VECTOR_CYPHER, embedding=embedding, top_k=top_k)
            rows = [dict(r) for r in records]
        logger.info("Vector retrieval executed", extra={"rows": len(rows)})
        return rows
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
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


def _run_wrrf_query(question: str, top_k: int) -> list[dict]:  # pragma: no cover
    """Execute WRRF hybrid search combining BM25, vector, and graph signals."""
    bm25_results = _run_bm25_query(question, top_k)
    vector_results = _run_vector_query(question, top_k)
    graph_results = _run_graph_query(question, top_k)

    if not (bm25_results or vector_results or graph_results):
        logger.warning("All retrieval signals returned empty results")
        return []

    fused = _wrrf_fusion(bm25_results, vector_results, graph_results, top_k)
    return fused


def _wrrf_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    graph_results: list[dict],
    top_k: int,
) -> list[dict]:  # pragma: no cover
    """Fuse results using Weighted Reciprocal Rank Fusion.

    Formula: score = sum(weight_i / (k + rank_i)) for each signal
    """
    from collections import defaultdict

    # Build rank maps for each signal
    bm25_ranks = {r["chunk_id"]: i + 1 for i, r in enumerate(bm25_results)}
    vector_ranks = {r["chunk_id"]: i + 1 for i, r in enumerate(vector_results)}
    graph_ranks = {r["chunk_id"]: i + 1 for i, r in enumerate(graph_results)}

    # Collect all chunk IDs
    all_chunk_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys()) | set(graph_ranks.keys())

    # Compute WRRF scores
    wrrf_scores = {}
    k = settings.wrrf_k
    w_bm25 = settings.wrrf_weight_bm25
    w_vector = settings.wrrf_weight_vector
    w_graph = settings.wrrf_weight_graph

    for chunk_id in all_chunk_ids:
        score = 0.0
        if chunk_id in bm25_ranks:
            score += w_bm25 / (k + bm25_ranks[chunk_id])
        if chunk_id in vector_ranks:
            score += w_vector / (k + vector_ranks[chunk_id])
        if chunk_id in graph_ranks:
            score += w_graph / (k + graph_ranks[chunk_id])
        wrrf_scores[chunk_id] = score

    # Build result list with metadata from best source
    result_map = defaultdict(dict)
    for r in bm25_results + vector_results + graph_results:
        chunk_id = r["chunk_id"]
        if chunk_id not in result_map:
            result_map[chunk_id] = {
                "page_title": r.get("page_title", ""),
                "page_url": r.get("page_url", ""),
                "page_id": r.get("page_id", ""),
                "chunk_id": chunk_id,
                "chunk_text": r.get("chunk_text", ""),
            }

    # Sort by WRRF score and return top_k
    sorted_results = sorted(
        [
            {**result_map[chunk_id], "score": wrrf_scores[chunk_id]}
            for chunk_id in wrrf_scores
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    logger.info(
        "WRRF fusion completed",
        extra={
            "bm25_count": len(bm25_results),
            "vector_count": len(vector_results),
            "graph_count": len(graph_results),
            "fused_count": len(sorted_results),
        },
    )

    return sorted_results[:top_k]


def _run_fallback_query(question: str, top_k: int) -> list[dict]:
    """Execute safe fallback query when LLM-generated Cypher fails."""
    with neo4j_client.session() as session:
        records = session.run(
            _HYBRID_FALLBACK_CYPHER,
            q=question,
            top_k=top_k,
        )
        rows = [dict(r) for r in records]
    logger.info("Fallback retrieval executed", extra={"rows": len(rows)})
    return rows


def _run_wrrf_query(question: str, top_k: int) -> list[dict]:
    """Execute WRRF hybrid search combining BM25, vector, and graph signals."""
    bm25_results = _run_bm25_query(question, top_k)
    vector_results = _run_vector_query(question, top_k)
    graph_results = _run_graph_query(question, top_k)

    if not (bm25_results or vector_results or graph_results):
        logger.warning("All retrieval signals returned empty results")
        return []

    fused = _wrrf_fusion(bm25_results, vector_results, graph_results, top_k)
    return fused


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
