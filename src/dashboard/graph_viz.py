"""Graph visualization API endpoints for the GraphPulse dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from src.infrastructure.neo4j_client import neo4j_client
from src.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/dashboard/api/graph", tags=["graph-viz"])


@router.get("/search-entities", response_class=JSONResponse)
def search_entities(
    q: str = Query(..., min_length=1, description="Search query for entity names"),
    limit: int = Query(10, ge=1, le=50, description="Max results to return"),
) -> JSONResponse:
    """Search entities by name prefix for autocomplete/typeahead."""
    try:
        with neo4j_client.session() as session:
            records = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($q)
                RETURN e.name AS name, e.type AS type, e.community_id AS community_id
                ORDER BY size(e.name)
                LIMIT $limit
                """,
                q=q,
                limit=limit,
            )
            rows = [dict(r) for r in records]

        return JSONResponse(content={"results": rows, "total": len(rows)})

    except Exception as exc:
        logger.warning("Entity search failed", extra={"error": str(exc)})
        return JSONResponse(content={"results": [], "total": 0, "error": str(exc)}, status_code=500)


@router.get("/search-pages", response_class=JSONResponse)
def search_pages(
    q: str = Query(..., min_length=1, description="Search query for page titles"),
    limit: int = Query(10, ge=1, le=50, description="Max results to return"),
) -> JSONResponse:
    """Search pages by title for autocomplete/typeahead."""
    try:
        with neo4j_client.session() as session:
            records = session.run(
                """
                MATCH (p:Page)
                WHERE toLower(p.title) CONTAINS toLower($q)
                RETURN p.title AS title, p.url AS url
                ORDER BY size(p.title)
                LIMIT $limit
                """,
                q=q,
                limit=limit,
            )
            rows = [dict(r) for r in records]

        return JSONResponse(content={"results": rows, "total": len(rows)})

    except Exception as exc:
        logger.warning("Page search failed", extra={"error": str(exc)})
        return JSONResponse(content={"results": [], "total": 0, "error": str(exc)}, status_code=500)


@router.get("/entity-neighborhood", response_class=JSONResponse)
def entity_neighborhood(
    name: str = Query(..., description="Entity name to explore"),
    hops: int = Query(1, ge=1, le=3, description="Number of hops"),
    limit: int = Query(20, ge=1, le=100, description="Max nodes to return"),
) -> JSONResponse:
    """Return entity neighborhood as nodes + edges for visualization."""
    try:
        with neo4j_client.session() as session:
            records = session.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($name)
                WITH e LIMIT 1
                CALL {
                    WITH e
                    MATCH (c:Chunk)-[:MENTIONS]->(e)
                    MATCH (c)-[:MENTIONS]->(co:Entity)
                    WHERE co <> e
                    RETURN DISTINCT co AS neighbor, 'co_mentioned' AS rel_type
                    LIMIT $limit
                }
                RETURN e.name AS center_name, e.type AS center_type,
                       collect(DISTINCT {
                           name: neighbor.name,
                           type: neighbor.type,
                           rel: rel_type
                       }) AS neighbors
                """,
                name=name,
                limit=limit,
            )
            rows = [dict(r) for r in records]

        if not rows or rows[0].get("center_name") is None:
            return JSONResponse(content={"nodes": [], "edges": [], "error": f"Entity '{name}' not found"})

        row = rows[0]
        nodes = [{"id": row["center_name"], "label": row["center_name"], "type": row["center_type"], "center": True}]
        edges = []

        for neighbor in row.get("neighbors", []):
            if not neighbor.get("name"):
                continue
            nodes.append({
                "id": neighbor["name"],
                "label": neighbor["name"],
                "type": neighbor.get("type"),
                "center": False,
            })
            edges.append({
                "source": row["center_name"],
                "target": neighbor["name"],
                "type": neighbor.get("rel", "co_mentioned"),
            })

        return JSONResponse(content={"nodes": nodes, "edges": edges})

    except Exception as exc:
        logger.warning("Graph viz entity neighborhood failed", extra={"error": str(exc)})
        return JSONResponse(content={"nodes": [], "edges": [], "error": str(exc)}, status_code=500)


@router.get("/page-graph", response_class=JSONResponse)
def page_graph(
    title: str = Query(..., description="Page title to explore"),
    depth: int = Query(1, ge=1, le=2, description="Link depth"),
) -> JSONResponse:
    """Return page link graph as nodes + edges."""
    try:
        with neo4j_client.session() as session:
            records = session.run(
                """
                MATCH (p:Page)
                WHERE toLower(p.title) CONTAINS toLower($title)
                WITH p LIMIT 1
                OPTIONAL MATCH (p)-[:LINKS_TO]->(linked:Page)
                WITH p, collect(DISTINCT {title: linked.title, url: linked.url})[..$limit] AS links
                RETURN p.title AS center_title, p.url AS center_url, links
                """,
                title=title,
                limit=30,
            )
            rows = [dict(r) for r in records]

        if not rows or rows[0].get("center_title") is None:
            return JSONResponse(content={"nodes": [], "edges": [], "error": f"Page '{title}' not found"})

        row = rows[0]
        nodes = [{"id": row["center_title"], "label": row["center_title"], "url": row["center_url"], "center": True}]
        edges = []

        for link in row.get("links", []):
            if not link.get("title"):
                continue
            nodes.append({
                "id": link["title"],
                "label": link["title"],
                "url": link.get("url"),
                "center": False,
            })
            edges.append({
                "source": row["center_title"],
                "target": link["title"],
                "type": "LINKS_TO",
            })

        return JSONResponse(content={"nodes": nodes, "edges": edges})

    except Exception as exc:
        logger.warning("Graph viz page graph failed", extra={"error": str(exc)})
        return JSONResponse(content={"nodes": [], "edges": [], "error": str(exc)}, status_code=500)


@router.get("/entity-types-distribution", response_class=JSONResponse)
def entity_types_distribution() -> JSONResponse:
    """Return entity type counts for chart visualization."""
    try:
        with neo4j_client.session() as session:
            records = session.run(
                """
                CALL {
                    MATCH (p:Person) RETURN 'Person' AS type, count(p) AS count
                } CALL {
                    MATCH (o:Organization) RETURN 'Organization' AS type2, count(o) AS count2
                } CALL {
                    MATCH (l:Location) RETURN 'Location' AS type3, count(l) AS count3
                } CALL {
                    MATCH (w:Work) RETURN 'Work' AS type4, count(w) AS count4
                }
                RETURN type, count, type2, count2, type3, count3, type4, count4
                """
            )
            record = records.single()

        if not record:
            return JSONResponse(content={"distribution": []})

        distribution = [
            {"type": "Person", "count": record["count"]},
            {"type": "Organization", "count": record["count2"]},
            {"type": "Location", "count": record["count3"]},
            {"type": "Work", "count": record["count4"]},
        ]

        return JSONResponse(content={"distribution": distribution})

    except Exception as exc:
        logger.warning("Graph viz entity distribution failed", extra={"error": str(exc)})
        return JSONResponse(content={"distribution": [], "error": str(exc)}, status_code=500)


@router.get("/shortest-path", response_class=JSONResponse)
def shortest_path(
    entity_a: str = Query(..., description="Start entity name"),
    entity_b: str = Query(..., description="End entity name"),
    max_hops: int = Query(4, ge=1, le=6, description="Maximum path length"),
) -> JSONResponse:
    """Find and return shortest path between two entities as nodes + edges."""
    try:
        max_rels = max_hops * 2
        with neo4j_client.session() as session:
            records = session.run(
                f"""
                MATCH (a:Entity)
                WHERE toLower(a.name) CONTAINS toLower($name_a)
                WITH a LIMIT 1
                MATCH (b:Entity)
                WHERE toLower(b.name) CONTAINS toLower($name_b)
                WITH a, b LIMIT 1
                MATCH path = shortestPath(
                    (a)-[:MENTIONS|HAS_CHUNK|LINKS_TO*..{max_rels}]-(b)
                )
                RETURN [n IN nodes(path) |
                    CASE
                        WHEN 'Entity' IN labels(n) THEN {{label: 'Entity', name: n.name, type: n.type}}
                        WHEN 'Chunk' IN labels(n) THEN {{label: 'Chunk', id: n.id}}
                        WHEN 'Page' IN labels(n) THEN {{label: 'Page', title: n.title}}
                        ELSE {{label: head(labels(n)), id: n.id}}
                    END
                ] AS path_nodes,
                [r IN relationships(path) | type(r)] AS rel_types,
                length(path) AS path_length
                """,
                name_a=entity_a,
                name_b=entity_b,
            )
            rows = [dict(r) for r in records]

        if not rows:
            return JSONResponse(content={
                "nodes": [],
                "edges": [],
                "path_length": None,
                "message": f"No path found between '{entity_a}' and '{entity_b}'",
            })

        row = rows[0]
        path_nodes = row.get("path_nodes", [])
        rel_types = row.get("rel_types", [])

        nodes = []
        for i, node in enumerate(path_nodes):
            label = node.get("label", "?")
            node_id = node.get("name") or node.get("title") or node.get("id") or f"node_{i}"
            nodes.append({
                "id": node_id,
                "label": node_id,
                "type": label,
                "entity_type": node.get("type"),
            })

        edges = []
        for i in range(len(rel_types)):
            if i < len(nodes) - 1:
                edges.append({
                    "source": nodes[i]["id"],
                    "target": nodes[i + 1]["id"],
                    "type": rel_types[i],
                })

        return JSONResponse(content={
            "nodes": nodes,
            "edges": edges,
            "path_length": row.get("path_length"),
        })

    except Exception as exc:
        logger.warning("Graph viz shortest path failed", extra={"error": str(exc)})
        return JSONResponse(content={"nodes": [], "edges": [], "error": str(exc)}, status_code=500)


@router.get("/community-overview", response_class=JSONResponse)
def community_overview(
    community_id: int = Query(..., ge=0, description="Community ID to visualize"),
    limit: int = Query(20, ge=1, le=50, description="Max entities to return"),
) -> JSONResponse:
    """Return community members and their co-mention edges for visualization."""
    try:
        with neo4j_client.session() as session:
            records = session.run(
                """
                MATCH (e:Entity)
                WHERE e.community_id = $cid
                WITH e LIMIT $limit
                WITH collect(e) AS members
                UNWIND members AS m
                OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(m)
                OPTIONAL MATCH (c)-[:MENTIONS]->(other:Entity)
                WHERE other IN members AND other <> m
                RETURN m.name AS name, m.type AS type,
                       collect(DISTINCT other.name) AS connected_to
                """,
                cid=community_id,
                limit=limit,
            )
            rows = [dict(r) for r in records]

        if not rows:
            return JSONResponse(content={
                "nodes": [], "edges": [], "community_id": community_id,
                "message": f"No entities found in community {community_id}",
            })

        nodes = []
        edges = []
        seen_edges: set[tuple[str, str]] = set()

        for row in rows:
            name = row.get("name")
            if not name:
                continue
            nodes.append({
                "id": name,
                "label": name,
                "type": row.get("type"),
                "community_id": community_id,
            })
            for target in row.get("connected_to", []):
                if not target:
                    continue
                edge_key = tuple(sorted([name, target]))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "source": name,
                        "target": target,
                        "type": "co_mentioned",
                    })

        return JSONResponse(content={
            "nodes": nodes,
            "edges": edges,
            "community_id": community_id,
            "member_count": len(nodes),
        })

    except Exception as exc:
        logger.warning("Graph viz community overview failed", extra={"error": str(exc)})
        return JSONResponse(content={"nodes": [], "edges": [], "error": str(exc)}, status_code=500)
