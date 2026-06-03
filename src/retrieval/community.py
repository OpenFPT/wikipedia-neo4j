"""Community-based retrieval using Louvain community summaries."""

from __future__ import annotations

from src.infrastructure.llm import embed_texts
from src.logging_utils import get_logger
from src.infrastructure.neo4j_client import neo4j_client

logger = get_logger(__name__)


def community_search(
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
