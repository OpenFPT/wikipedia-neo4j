"""BM25 fulltext search over Neo4j chunk_text_ft index."""

from __future__ import annotations

from src.logging_utils import get_logger
from src.infrastructure.neo4j_client import neo4j_client

logger = get_logger(__name__)

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


def run_bm25_query(question: str, top_k: int) -> list[dict]:  # pragma: no cover
    """Execute BM25 fulltext search."""
    with neo4j_client.session() as session:
        records = session.run(_BM25_CYPHER, q=question, top_k=top_k)
        rows = [dict(r) for r in records]
    logger.info("BM25 retrieval executed", extra={"rows": len(rows)})
    return rows
