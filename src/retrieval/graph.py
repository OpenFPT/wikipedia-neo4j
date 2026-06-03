"""Graph-based entity search and multi-hop link expansion."""

from __future__ import annotations

from src.logging_utils import get_logger
from src.infrastructure.neo4j_client import neo4j_client

logger = get_logger(__name__)

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


def run_graph_query(question: str, top_k: int) -> list[dict]:  # pragma: no cover
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


def graph_search(query: str, top_k: int) -> list[dict]:
    """Entity-based graph traversal: find entities mentioned in query, return their chunks.

    Uses the entity_alias_ft fulltext index to match query terms against entity names,
    then traverses MENTIONS edges to retrieve associated chunks.

    Args:
        query: Natural language query string.
        top_k: Maximum number of results to return.

    Returns:
        List of dicts with keys: page_title, page_url, page_id, chunk_id, chunk_text, graph_score.
    """
    try:
        with neo4j_client.session() as session:
            records = session.run(_GRAPH_CYPHER, q=query, top_k=top_k)
            rows = [dict(r) for r in records]
        logger.info("_graph_search executed", extra={"rows": len(rows)})
        return rows
    except Exception as e:
        logger.warning(f"_graph_search failed: {e}")
        return []


def expand_via_links(page_ids: list[str], question: str, top_k: int) -> list[dict]:
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
