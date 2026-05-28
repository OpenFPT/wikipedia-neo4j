"""MCP tool definitions for the WikiGraphRAG system."""

from __future__ import annotations

import re

from src.logging_utils import get_logger
from src.neo4j_client import neo4j_client
from src.retrieve import hybrid_retrieve

logger = get_logger(__name__)

_BLOCKED_KEYWORDS = [
    "create", "merge", "delete", "detach", "set",
    "remove", "drop", "load csv", "apoc.periodic", "call dbms",
]


def _validate_readonly_cypher(cypher: str) -> None:
    """Validate that a Cypher query is read-only. Raises ValueError if not."""
    raw = (cypher or "").strip()
    if not raw:
        raise ValueError("Cypher query is empty")
    trimmed = raw[:-1] if raw.endswith(";") else raw
    if ";" in trimmed:
        raise ValueError("Multiple statements not allowed")
    stripped = re.sub(r"//[^\n]*", " ", raw)
    stripped = re.sub(r"/\*.*?\*/", " ", stripped, flags=re.DOTALL)
    lowered = re.sub(r"\s+", " ", stripped.lower())
    for kw in _BLOCKED_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            raise ValueError(f"Write operation detected: {kw}")


def register_tools(mcp) -> None:  # noqa: ANN001
    """Register all GraphRAG tools on the given FastMCP instance."""

    @mcp.tool()
    def search_knowledge_base(
        question: str,
        top_k: int = 5,
        method: str = "hybrid",
    ) -> dict:
        """Use for factual questions needing 1-2 facts from Vietnamese Wikipedia.

        Examples: 'Ai sáng lập Đảng Cộng sản Việt Nam?', 'Hà Nội có bao nhiêu quận?'
        Do NOT use for questions requiring reasoning across multiple facts.
        Returns ranked text passages with source page links.

        Args:
            question: Question in Vietnamese.
            top_k: Number of results (1-20, default 5).
            method: Retrieval method — "hybrid" (default), "bm25", "vector", or "graph".
        """
        logger.info("MCP tool: search_knowledge_base", extra={"question": question})
        try:
            results = hybrid_retrieve(question, top_k=top_k)
            return {
                "question": question,
                "method": method,
                "results": [
                    {
                        "chunk_id": r.get("chunk_id", ""),
                        "text": r.get("chunk_text", "")[:500],
                        "score": round(r.get("score", 0.0), 4),
                        "page_title": r.get("page_title", ""),
                        "page_url": r.get("page_url", ""),
                    }
                    for r in results
                ],
                "total": len(results),
            }
        except Exception as e:
            logger.exception("search_knowledge_base failed")
            return {"error": str(e), "question": question, "results": [], "total": 0}

    @mcp.tool()
    def explore_entity(entity_name: str, depth: int = 1) -> dict:
        """Use when you know an entity name and want to see what connects to it.

        Examples: explore 'Hồ Chí Minh' to find related people, places, events.
        Do NOT use for searching — use search_knowledge_base first to find entity names.

        Args:
            entity_name: Entity name in Vietnamese (e.g. "Hồ Chí Minh").
            depth: Traversal depth, 1 or 2 hops (default 1).
        """
        logger.info("MCP tool: explore_entity", extra={"entity": entity_name})
        depth = min(max(depth, 1), 2)
        try:
            with neo4j_client.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity {name: $name})
                    OPTIONAL MATCH (e)-[r]-(connected:Entity)
                    RETURN e.name AS entity, e.type AS entity_type,
                           type(r) AS rel_type, connected.name AS connected_name,
                           connected.type AS connected_type
                    LIMIT 50
                    """,
                    name=entity_name,
                )
                neighbors = [dict(record) for record in result]
            return {
                "entity": entity_name,
                "depth": depth,
                "neighbors": neighbors,
                "count": len(neighbors),
            }
        except Exception as e:
            logger.exception("explore_entity failed")
            return {"error": str(e), "entity": entity_name, "neighbors": [], "count": 0}

    @mcp.tool()
    def find_path(entity_a: str, entity_b: str, max_hops: int = 5) -> dict:
        """Use for questions asking HOW two things are connected or related.

        Examples: 'Mối quan hệ giữa Phạm Văn Đồng và Hồ Chí Minh?'
        Requires knowing both entity names — use search_knowledge_base first if unsure.

        Args:
            entity_a: Starting entity name (Vietnamese).
            entity_b: Target entity name (Vietnamese).
            max_hops: Maximum path length (default 5, max 6).
        """
        logger.info("MCP tool: find_path", extra={"from": entity_a, "to": entity_b})
        max_hops = min(max(max_hops, 2), 6)
        try:
            with neo4j_client.session() as session:
                result = session.run(
                    f"""
                    MATCH path = shortestPath(
                        (a:Entity {{name: $a}})-[*..{max_hops}]-(b:Entity {{name: $b}})
                    )
                    RETURN [n IN nodes(path) | n.name] AS nodes,
                           [r IN relationships(path) | type(r)] AS relations
                    """,
                    a=entity_a,
                    b=entity_b,
                )
                record = result.single()
            if record:
                return {
                    "from": entity_a,
                    "to": entity_b,
                    "path": record["nodes"],
                    "relations": record["relations"],
                    "hops": len(record["relations"]),
                }
            return {"from": entity_a, "to": entity_b, "path": None, "message": "No path found"}
        except Exception as e:
            logger.exception("find_path failed")
            return {"error": str(e), "from": entity_a, "to": entity_b, "path": None}

    @mcp.tool()
    def get_community_summary(topic: str) -> dict:
        """Use for broad topic overviews before diving into specifics.

        Returns a summary of related concepts clustered around a topic.
        Examples: get overview of 'Chiến tranh Việt Nam' or 'Văn học Việt Nam'.
        Do NOT use for specific factual questions — use search_knowledge_base instead.

        Args:
            topic: Topic or entity name to find community for.
        """
        logger.info("MCP tool: get_community_summary", extra={"topic": topic})
        try:
            from src.community import (
                get_community_for_entity,
                get_community_summary as _get_summary,
            )

            community_id = get_community_for_entity(topic)
            if community_id is None:
                return {"topic": topic, "summary": None, "message": "No community found for topic"}
            summary = _get_summary(community_id)
            return {"topic": topic, "community_id": community_id, "summary": summary}
        except Exception as e:
            logger.exception("get_community_summary failed")
            return {"error": str(e), "topic": topic, "summary": None}

    @mcp.tool()
    def answer_question(question: str) -> dict:
        """Use for complex questions requiring reasoning across multiple facts.

        Automatically decomposes multi-hop questions, retrieves from multiple sources,
        and synthesizes an answer with citations.
        Do NOT use for simple factual lookups — use search_knowledge_base instead.

        Args:
            question: Question in Vietnamese.
        """
        logger.info("MCP tool: answer_question", extra={"question": question})
        try:
            from src.agent import run_agent_scaled

            result = run_agent_scaled(question)
            return {
                "question": question,
                "answer": result.answer,
                "citations": result.citations,
                "retrieval_tier": result.retrieval_tier,
            }
        except Exception as e:
            logger.exception("answer_question failed")
            return {"error": str(e), "question": question, "answer": "", "citations": []}

    @mcp.tool()
    def get_graph_stats() -> dict:
        """Get statistics about the knowledge graph.

        Returns counts of pages, chunks, entities, and relationships.
        Useful for understanding the scope and coverage of the knowledge base.
        """
        logger.info("MCP tool: get_graph_stats")
        try:
            with neo4j_client.session() as session:
                result = session.run("""
                    CALL {
                        MATCH (p:Page) RETURN count(p) AS pages
                    }
                    CALL {
                        MATCH (c:Chunk) RETURN count(c) AS chunks
                    }
                    CALL {
                        MATCH (e:Entity) RETURN count(e) AS entities
                    }
                    RETURN pages, chunks, entities
                """)
                record = result.single()
            if record:
                return {
                    "pages": record["pages"],
                    "chunks": record["chunks"],
                    "entities": record["entities"],
                }
            return {"pages": 0, "chunks": 0, "entities": 0}
        except Exception as e:
            logger.exception("get_graph_stats failed")
            return {"error": str(e), "pages": 0, "chunks": 0, "entities": 0}

    @mcp.tool()
    def list_entity_types() -> dict:
        """List all entity types in the knowledge graph with counts.

        Returns entity types (Person, Organization, Location, Work) and their counts.
        Useful for understanding what kinds of entities are available to query.
        """
        logger.info("MCP tool: list_entity_types")
        try:
            with neo4j_client.session() as session:
                result = session.run("""
                    MATCH (e:Entity)
                    RETURN e.type AS type, count(*) AS count
                    ORDER BY count DESC
                """)
                types = [dict(r) for r in result]
            return {"entity_types": types, "total_types": len(types)}
        except Exception as e:
            logger.exception("list_entity_types failed")
            return {"error": str(e), "entity_types": [], "total_types": 0}

    @mcp.tool()
    def source_trace(chunk_id: str) -> dict:
        """Use after search_knowledge_base to get more context around a passage.

        Given a chunk_id from search results, returns the full source page title
        and neighboring chunks for surrounding context.
        Use when a search result is relevant but you need more detail to answer.

        Args:
            chunk_id: Chunk ID from search results.
        """
        logger.info("MCP tool: source_trace", extra={"chunk_id": chunk_id})
        try:
            with neo4j_client.session() as session:
                result = session.run(
                    """
                    MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk {id: $chunk_id})
                    OPTIONAL MATCH (p)-[:HAS_CHUNK]->(sibling:Chunk)
                    WHERE abs(sibling.sequence_number - c.sequence_number) <= 1
                    RETURN p.title AS page_title, p.url AS page_url,
                           c.text AS chunk_text, c.sequence_number AS seq,
                           collect(DISTINCT {
                               id: sibling.id,
                               text: sibling.text,
                               seq: sibling.sequence_number
                           }) AS neighbors
                    """,
                    chunk_id=chunk_id,
                )
                record = result.single()
            if record:
                return {
                    "chunk_id": chunk_id,
                    "page_title": record["page_title"],
                    "page_url": record["page_url"],
                    "chunk_text": record["chunk_text"],
                    "sequence_number": record["seq"],
                    "neighbors": sorted(record["neighbors"], key=lambda x: x["seq"] or 0),
                }
            return {"chunk_id": chunk_id, "error": "Chunk not found"}
        except Exception as e:
            logger.exception("source_trace failed")
            return {"error": str(e), "chunk_id": chunk_id}

    @mcp.tool()
    def kg_query(cypher: str, params: dict | None = None) -> dict:
        """Execute a read-only Cypher query against the knowledge graph.

        Use this for flexible graph queries when pre-built tools don't cover your need.

        Schema:
        - Page(id, title, url, summary)
        - Chunk(id, text, sequence_number, embedding)
        - Entity(id, name, type) with labels: Person, Organization, Location, Work
        - Community(id, embedding)
        - (Page)-[:HAS_CHUNK]->(Chunk)
        - (Chunk)-[:MENTIONS]->(Entity) + typed: MENTIONS_PERSON, MENTIONS_ORG, MENTIONS_LOCATION, MENTIONS_WORK
        - (Page)-[:LINKS_TO]->(Page)
        - Relation edges: FOUNDED_BY, LOCATED_IN, BORN_IN, MEMBER_OF, PART_OF, CREATED_BY
        - Fulltext indexes: chunk_text_ft (Chunk.text), page_title_ft (Page.title+summary), entity_alias_ft (Entity.name+aliases)
        - Vector index: chunk_embedding_idx (Chunk.embedding, 1024-dim cosine)

        Only read operations allowed (MATCH/RETURN/WITH/WHERE/ORDER BY/LIMIT/CALL).

        Args:
            cypher: A read-only Cypher query string.
            params: Optional query parameters as a dict.
        """
        logger.info("MCP tool: kg_query", extra={"cypher": cypher[:200]})
        try:
            _validate_readonly_cypher(cypher)
        except ValueError as e:
            return {"error": str(e), "cypher": cypher, "results": [], "total": 0}

        try:
            with neo4j_client.session() as session:
                result = session.run(cypher, **(params or {}))
                rows = [dict(r) for r in result]
            return {
                "cypher": cypher,
                "results": rows[:50],
                "total": len(rows),
                "truncated": len(rows) > 50,
            }
        except Exception as e:
            logger.exception("kg_query failed")
            return {"error": str(e), "cypher": cypher, "results": [], "total": 0}

    @mcp.resource("graph://schema")
    def graph_schema() -> str:
        """The Neo4j knowledge graph schema for Vietnamese Wikipedia."""
        return (
            "Node labels: Page(id, title, url, summary), "
            "Chunk(id, text, sequence_number, embedding), "
            "Entity(id, name, type) with optional labels Person/Organization/Location/Work.\n"
            "Relationships: (Page)-[:HAS_CHUNK]->(Chunk), (Chunk)-[:MENTIONS]->(Entity), "
            "(Page)-[:LINKS_TO]->(Page).\n"
            "Typed mention edges: MENTIONS_PERSON, MENTIONS_ORG, MENTIONS_LOCATION, MENTIONS_WORK.\n"
            "Indexes: fulltext 'chunk_text_ft' on Chunk.text, "
            "fulltext 'page_title_ft' on Page.title+summary, "
            "vector 'chunk_embedding_idx' on Chunk.embedding (1024-dim)."
        )
