"""Dense vector similarity search over Neo4j chunk_embedding_idx."""

from __future__ import annotations

from src.config import settings
from src.infrastructure.llm import embed_texts
from src.logging_utils import get_logger
from src.infrastructure.neo4j_client import neo4j_client

logger = get_logger(__name__)

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


def run_vector_query(
    question: str, top_k: int, *, query_embedding: list[float] | None = None
) -> list[dict]:  # pragma: no cover
    """Execute vector similarity search."""
    if settings.neo4j_use_search_clause:
        return run_vector_query_cypher25(question, top_k, query_embedding=query_embedding)
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


def run_vector_query_cypher25(
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


def vector_search(query_embedding: list[float], top_k: int) -> list[dict]:
    """Query Neo4j vector index on Chunk.embedding directly.

    Args:
        query_embedding: Pre-computed embedding vector for the query.
        top_k: Maximum number of results to return.

    Returns:
        List of dicts with keys: page_title, page_url, page_id, chunk_id, chunk_text, vector_score.
    """
    if not query_embedding:
        return []
    try:
        if settings.neo4j_use_search_clause:
            with neo4j_client.session() as session:
                records = session.run(
                    _VECTOR_SEARCH_CYPHER25,
                    query_embedding=query_embedding,
                    top_k=top_k,
                )
                rows = [dict(r) for r in records]
        else:
            with neo4j_client.session() as session:
                records = session.run(
                    _VECTOR_CYPHER,
                    embedding=query_embedding,
                    top_k=top_k,
                )
                rows = [dict(r) for r in records]
        logger.info("_vector_search executed", extra={"rows": len(rows)})
        return rows
    except Exception as e:
        logger.warning(f"_vector_search failed: {e}")
        return []
