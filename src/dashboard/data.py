"""Data fetching functions for the GraphPulse dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import settings
from src.dashboard.query_log import query_log
from src.logging_utils import get_logger

logger = get_logger(__name__)


def fetch_graph_stats() -> dict:
    """Fetch node and relationship counts from Neo4j.

    Returns a dict with counts, or zeros with an error flag on failure.
    """
    try:
        from src.infrastructure.neo4j_client import neo4j_client

        with neo4j_client.session() as session:
            result = session.run(
                """
                CALL {
                    MATCH (p:Page) RETURN count(p) AS pages
                } CALL {
                    MATCH (c:Chunk) RETURN count(c) AS chunks
                } CALL {
                    MATCH (e:Entity) RETURN count(e) AS entities
                } CALL {
                    MATCH (p:Person) RETURN count(p) AS persons
                } CALL {
                    MATCH (o:Organization) RETURN count(o) AS orgs
                } CALL {
                    MATCH (l:Location) RETURN count(l) AS locations
                } CALL {
                    MATCH (w:Work) RETURN count(w) AS works
                } CALL {
                    MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) AS has_chunk_rels
                } CALL {
                    MATCH ()-[r:MENTIONS]->() RETURN count(r) AS mention_rels
                } CALL {
                    MATCH ()-[r:LINKS_TO]->() RETURN count(r) AS links_to_rels
                }
                RETURN pages, chunks, entities, persons, orgs, locations, works,
                       has_chunk_rels, mention_rels, links_to_rels
                """
            )
            record = result.single()
            if record:
                return {
                    "pages": record["pages"],
                    "chunks": record["chunks"],
                    "entities": record["entities"],
                    "persons": record["persons"],
                    "orgs": record["orgs"],
                    "locations": record["locations"],
                    "works": record["works"],
                    "has_chunk_rels": record["has_chunk_rels"],
                    "mention_rels": record["mention_rels"],
                    "links_to_rels": record["links_to_rels"],
                    "total_rels": (
                        record["has_chunk_rels"]
                        + record["mention_rels"]
                        + record["links_to_rels"]
                    ),
                    "available": True,
                }
    except Exception as exc:
        logger.warning("Failed to fetch graph stats", extra={"error": str(exc)})

    return {
        "pages": None,
        "chunks": None,
        "entities": None,
        "persons": None,
        "orgs": None,
        "locations": None,
        "works": None,
        "has_chunk_rels": None,
        "mention_rels": None,
        "links_to_rels": None,
        "total_rels": None,
        "available": False,
    }


def fetch_recent_queries(n: int = 20) -> list[dict]:
    """Return recent query log entries."""
    return query_log.recent(n)


def fetch_signal_breakdown() -> dict | None:
    """Return signal scores from the most recent query, or None."""
    entry = query_log.latest()
    if entry and entry.signal_scores:
        return {
            "scores": entry.signal_scores,
            "question": entry.question,
            "timestamp": entry.timestamp,
        }
    return None


def fetch_eval_metrics() -> dict | None:
    """Read evaluation metrics from data/eval_results.json if it exists."""
    eval_path = Path("data/eval_results.json")
    if not eval_path.exists():
        return None
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "timestamp": data.get("timestamp", ""),
            "total": data.get("total", 0),
            "context_hit_rate": data.get("context_hit_rate", 0.0),
            "mrr": data.get("mrr", 0.0),
            "rerank_context_hit_rate": data.get("rerank_context_hit_rate", 0.0),
            "rerank_mrr": data.get("rerank_mrr", 0.0),
            "avg_latency_ms": data.get("avg_latency_ms", 0),
            "available": True,
        }
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read eval results", extra={"error": str(exc)})
        return None


def fetch_wrrf_weights() -> dict:
    """Return configured WRRF weights from settings."""
    return {
        "bm25": settings.wrrf_weight_bm25,
        "vector": settings.wrrf_weight_vector,
        "graph": settings.wrrf_weight_graph,
        "community": settings.wrrf_weight_community,
    }


def fetch_qa_stats() -> dict:
    """Fetch multi-hop QA benchmark statistics from Neo4j.

    Returns question-type distribution, hop-count histogram, and
    answerable/unanswerable ratio, or zeros with an error flag on failure.
    """
    try:
        from src.infrastructure.neo4j_client import neo4j_client

        with neo4j_client.session() as session:
            result = session.run(
                """
                CALL {
                    MATCH (q:Question) RETURN count(q) AS total_questions
                } CALL {
                    MATCH (q:Question) WHERE q.is_impossible = true RETURN count(q) AS unanswerable
                } CALL {
                    MATCH ()-[r:SUPPORTED_BY]->() RETURN count(r) AS supported_by_rels
                } CALL {
                    MATCH ()-[r:BRIDGES]->() RETURN count(r) AS bridges_rels
                } CALL {
                    MATCH (q:Question)
                    RETURN q.type AS type, count(q) AS cnt
                    ORDER BY cnt DESC
                }
                WITH total_questions, unanswerable, supported_by_rels, bridges_rels,
                     collect({type: type, count: cnt}) AS by_type
                RETURN total_questions, unanswerable, supported_by_rels, bridges_rels, by_type
                """
            )
            record = result.single()
            if record:
                total = record["total_questions"] or 0
                unanswerable = record["unanswerable"] or 0
                return {
                    "total_questions": total,
                    "answerable": total - unanswerable,
                    "unanswerable": unanswerable,
                    "supported_by_rels": record["supported_by_rels"],
                    "bridges_rels": record["bridges_rels"],
                    "by_type": [row for row in record["by_type"] if row.get("type")],
                    "available": True,
                }
    except Exception as exc:
        logger.warning("Failed to fetch QA stats", extra={"error": str(exc)})

    return {
        "total_questions": None,
        "answerable": None,
        "unanswerable": None,
        "supported_by_rels": None,
        "bridges_rels": None,
        "by_type": [],
        "available": False,
    }


def fetch_top_bridged_pages(limit: int = 10) -> list[dict]:
    """Return the pages most frequently spanned by multi-hop questions."""
    try:
        from src.infrastructure.neo4j_client import neo4j_client

        with neo4j_client.session() as session:
            result = session.run(
                """
                MATCH (q:Question)-[:BRIDGES]->(p:Page)
                WITH p, count(q) AS bridge_count
                ORDER BY bridge_count DESC
                LIMIT $limit
                RETURN p.title AS title, bridge_count
                """,
                limit=limit,
            )
            return [dict(r) for r in result]
    except Exception as exc:
        logger.warning("Failed to fetch top bridged pages", extra={"error": str(exc)})
        return []
