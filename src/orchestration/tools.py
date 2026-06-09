"""Agent tools for the ReAct agent: kg_schema, kg_query, text_search, get_passage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config import settings
from src.logging_utils import get_logger
from src.infrastructure.neo4j_client import neo4j_client

logger = get_logger(__name__)

COLLECTION_NAME = "viwiki_paragraphs"


@dataclass
class ToolResult:
    """Standard result from any agent tool."""

    tool_name: str
    success: bool
    data: dict | list | str
    error: str | None = None


def kg_schema() -> ToolResult:
    """Return the current Neo4j knowledge graph schema (node labels, relationship types, properties)."""
    try:
        with neo4j_client.session() as session:
            labels = [r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label")]
            rel_types = [r["relationshipType"] for r in session.run(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            )]
            props = [dict(r) for r in session.run(
                "CALL db.schema.nodeTypeProperties() YIELD nodeType, propertyName, propertyTypes "
                "RETURN nodeType, propertyName, propertyTypes LIMIT 100"
            )]

        schema = {
            "node_labels": labels,
            "relationship_types": rel_types,
            "node_properties": props,
        }
        return ToolResult(tool_name="kg_schema", success=True, data=schema)
    except Exception as e:
        return ToolResult(tool_name="kg_schema", success=False, data={}, error=str(e))


def kg_query(cypher: str, params: dict | None = None) -> ToolResult:
    """Execute a read-only Cypher query against the knowledge graph."""
    cypher_upper = cypher.strip().upper()
    write_keywords = ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP", "DETACH"]
    for kw in write_keywords:
        if kw in cypher_upper and "IF NOT EXISTS" not in cypher_upper:
            return ToolResult(
                tool_name="kg_query",
                success=False,
                data=[],
                error=f"Write operation detected ({kw}). Only read queries allowed.",
            )

    try:
        with neo4j_client.session() as session:
            result = session.run(cypher, **(params or {}))
            rows = [dict(r) for r in result]

        return ToolResult(tool_name="kg_query", success=True, data=rows[:50])
    except Exception as e:
        return ToolResult(tool_name="kg_query", success=False, data=[], error=str(e))


def text_search(
    query: str,
    top_k: int = 5,
    article_filter: str | None = None,
) -> ToolResult:
    """Hybrid search: fulltext in Neo4j + vector similarity in Qdrant."""
    results: list[dict[str, Any]] = []

    # Neo4j fulltext search
    try:
        with neo4j_client.session() as session:
            records = session.run(
                """
                CALL db.index.fulltext.queryNodes('paragraph_text_ft', $query)
                YIELD node, score
                MATCH (a:Article)-[:HAS_PARAGRAPH]->(node)
                RETURN node.id AS paragraph_id,
                       a.title AS article_title,
                       node.text AS text,
                       score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                parameters={"query": query, "top_k": top_k},
            )
            for r in records:
                results.append({
                    "paragraph_id": r["paragraph_id"],
                    "article_title": r["article_title"],
                    "text": r["text"][:500],
                    "score": r["score"],
                    "source": "fulltext",
                })
    except Exception as e:
        logger.warning("Fulltext search failed", extra={"error": str(e)})

    # Qdrant vector search (if available)
    try:
        qdrant = QdrantClient(url=settings.qdrant_url)
        search_filter = None
        if article_filter:
            search_filter = Filter(must=[
                FieldCondition(key="article_title", match=MatchValue(value=article_filter))
            ])

        # Use Qdrant's built-in query if embeddings are pre-computed
        # For now, we do a scroll with filter as placeholder until embeddings are generated
        scroll_result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=search_filter,
            limit=top_k,
        )
        for point in scroll_result[0]:
            payload = point.payload or {}
            results.append({
                "paragraph_id": payload.get("paragraph_id", str(point.id)),
                "article_title": payload.get("article_title", ""),
                "text": payload.get("text", "")[:500],
                "score": 0.0,
                "source": "vector",
            })
    except Exception as e:
        logger.debug("Qdrant search skipped", extra={"error": str(e)})

    # Deduplicate by paragraph_id, keep highest score
    seen: dict[str, dict[str, Any]] = {}
    for item in results:
        pid = str(item["paragraph_id"])
        if pid not in seen or item["score"] > seen[pid]["score"]:  # type: ignore[operator]
            seen[pid] = item

    final = sorted(seen.values(), key=lambda x: float(x["score"]), reverse=True)[:top_k]
    return ToolResult(tool_name="text_search", success=True, data=final)


def get_passage(paragraph_id: str) -> ToolResult:
    """Retrieve the full text of a specific paragraph by ID."""
    try:
        with neo4j_client.session() as session:
            result = session.run(
                """
                MATCH (p:Paragraph {id: $id})
                OPTIONAL MATCH (a:Article)-[:HAS_PARAGRAPH]->(p)
                RETURN p.id AS paragraph_id,
                       p.text AS text,
                       p.paragraph_index AS paragraph_index,
                       a.title AS article_title,
                       a.url AS article_url
                """,
                id=paragraph_id,
            )
            record = result.single()

        if record is None:
            return ToolResult(
                tool_name="get_passage",
                success=False,
                data={},
                error=f"Paragraph not found: {paragraph_id}",
            )

        return ToolResult(tool_name="get_passage", success=True, data=dict(record))
    except Exception as e:
        return ToolResult(tool_name="get_passage", success=False, data={}, error=str(e))


# Tool registry for the ReAct agent
AGENT_TOOLS = {
    "kg_schema": kg_schema,
    "kg_query": kg_query,
    "text_search": text_search,
    "get_passage": get_passage,
}
