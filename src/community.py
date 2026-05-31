"""Community-based retrieval for GraphRAG.

Provides lookup of community membership, pre-generated summaries,
and retrieval of chunks belonging to relevant communities.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from src.logging_utils import get_logger
from src.infrastructure.neo4j_client import neo4j_client

logger = get_logger(__name__)

_SUMMARIES_PATH = Path("data/communities/summaries.jsonl")

# Lazy-loaded singleton for community summaries
_summaries: dict[int, dict] | None = None
_summaries_lock = Lock()


def _load_summaries() -> dict[int, dict]:
    """Load community summaries from JSONL file. Lazy singleton."""
    global _summaries
    if _summaries is not None:
        return _summaries

    with _summaries_lock:
        # Double-check after acquiring lock
        if _summaries is not None:
            return _summaries

        loaded: dict[int, dict] = {}
        if _SUMMARIES_PATH.exists():
            with open(_SUMMARIES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    # community_id format: "community_00042" -> extract int
                    cid_raw = record.get("community_id", "")
                    if isinstance(cid_raw, str) and cid_raw.startswith("community_"):
                        cid = int(cid_raw.split("_")[1])
                    elif isinstance(cid_raw, int):
                        cid = cid_raw
                    else:
                        continue
                    loaded[cid] = {
                        "community_id": cid,
                        "summary": record.get("summary", ""),
                        "member_count": record.get("member_count", 0),
                        "top_entities": record.get("top_entities", []),
                        "entity_ids": record.get("entity_ids", []),
                    }
            logger.info(f"Loaded {len(loaded)} community summaries from {_SUMMARIES_PATH}")
        else:
            logger.warning(f"Community summaries file not found: {_SUMMARIES_PATH}")

        _summaries = loaded
        return _summaries


def reload_summaries() -> None:
    """Force reload of community summaries from disk."""
    global _summaries
    with _summaries_lock:
        _summaries = None
    _load_summaries()


def get_community_for_entity(entity_name: str) -> int | None:
    """Look up the community ID for a given entity name.

    Queries Neo4j for the community_id property on the Entity node.
    Returns None if the entity is not found or has no community assignment.
    """
    with neo4j_client.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)
            WHERE e.name = $name
            RETURN e.community_id AS community_id
            LIMIT 1
            """,
            name=entity_name,
        )
        record = result.single()
        if record and record["community_id"] is not None:
            return int(record["community_id"])
    return None


def get_community_summary(community_id: int) -> str:
    """Get the pre-generated summary for a community.

    Falls back to Neo4j Community node if JSONL file is unavailable.
    """
    summaries = _load_summaries()

    # Try JSONL cache first
    if community_id in summaries:
        return summaries[community_id]["summary"]

    # Fallback: query Community node in Neo4j
    community_str_id = f"community_{community_id:05d}"
    with neo4j_client.session() as session:
        result = session.run(
            """
            MATCH (cm:Community {id: $cid})
            RETURN cm.summary AS summary
            """,
            cid=community_str_id,
        )
        record = result.single()
        if record and record["summary"]:
            return record["summary"]

    return ""


def retrieve_by_community(query: str, top_k: int = 5) -> list[dict]:
    """Find relevant communities via entity matching and return their chunks.

    Strategy:
    1. Use fulltext search to find entities matching the query.
    2. Look up their community_id.
    3. Retrieve chunks from entities in the same communities.
    4. Deduplicate and rank by relevance (number of community entity mentions).

    Returns list of dicts with keys: page_title, page_url, chunk_id, chunk_text, score, community_id.
    """
    # Step 1: Find entities matching the query via fulltext
    matched_communities: dict[int, float] = {}

    with neo4j_client.session() as session:
        entity_results = session.run(
            """
            CALL db.index.fulltext.queryNodes('entity_alias_ft', $q)
            YIELD node, score
            WHERE node.community_id IS NOT NULL
            RETURN node.community_id AS community_id, score
            ORDER BY score DESC
            LIMIT 20
            """,
            q=query,
        )
        for r in entity_results:
            cid = int(r["community_id"])
            # Accumulate scores per community
            matched_communities[cid] = matched_communities.get(cid, 0.0) + r["score"]

    if not matched_communities:
        logger.debug("No communities matched for query", extra={"query": query[:100]})
        return []

    # Step 2: Rank communities by accumulated score, take top ones
    ranked = sorted(matched_communities.items(), key=lambda x: x[1], reverse=True)
    top_community_ids = [cid for cid, _ in ranked[:3]]

    # Step 3: Retrieve chunks from these communities
    chunks: list[dict] = []
    seen_chunk_ids: set[str] = set()

    with neo4j_client.session() as session:
        for cid in top_community_ids:
            community_str_id = f"community_{cid:05d}"
            results = session.run(
                """
                MATCH (cm:Community {id: $cid})-[:HAS_MEMBER]->(e:Entity)<-[:MENTIONS]-(c:Chunk)
                MATCH (p:Page)-[:HAS_CHUNK]->(c)
                RETURN DISTINCT
                    p.title AS page_title,
                    p.url AS page_url,
                    c.id AS chunk_id,
                    c.text AS chunk_text,
                    cm.id AS community_id
                LIMIT $limit
                """,
                cid=community_str_id,
                limit=top_k * 2,
            )
            community_score = matched_communities.get(cid, 0.0)
            for r in results:
                chunk_id = r["chunk_id"]
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
                chunks.append(
                    {
                        "page_title": r["page_title"],
                        "page_url": r["page_url"],
                        "chunk_id": chunk_id,
                        "chunk_text": r["chunk_text"],
                        "score": community_score,
                        "community_id": cid,
                    }
                )

    # Sort by score descending and limit
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks[:top_k]
