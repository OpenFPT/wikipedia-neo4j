"""Core ReAct agent loop: think → act → observe."""

from __future__ import annotations

import json
import re
from typing import Literal

from src.logging_utils import get_logger
from src.infrastructure.neo4j_client import neo4j_client
from src.retrieval.fusion import QueryResult, QueryTrace, QueryTraceStep, hybrid_retrieve
from src.retrieval.reranker import rerank

from src.orchestration._parsing import parse_agent_response
from src.orchestration.complexity import detect_complexity, COMPLEXITY_THRESHOLD
from src.orchestration.decomposition import decompose_question, agent_query_with_decomposition

logger = get_logger(__name__)

MAX_ITERATIONS = 6
_ALLOWED_SCHEMA_IDENTIFIERS = {
    "Page", "Chunk", "Entity", "Person", "Organization", "Location", "Work", "Community", "Question",
    "HAS_CHUNK", "MENTIONS", "LINKS_TO", "FOUNDED_BY", "LOCATED_IN", "BORN_IN", "MEMBER_OF", "PART_OF", "CREATED_BY",
}
_ALLOWED_PROPERTY_KEYS = {
    "id", "title", "url", "summary", "text", "sequence_number", "embedding",
    "name", "type", "source", "aliases",
}

_GRAPH_SCHEMA = (
    "Node labels: Page(id, title, url, summary), Chunk(id, text, sequence_number, embedding), "
    "Entity(id, name, type) with optional labels Person/Organization/Location/Work.\n"
    "Relationships: (Page)-[:HAS_CHUNK]->(Chunk), (Chunk)-[:MENTIONS]->(Entity), "
    "(Page)-[:LINKS_TO]->(Page).\n"
    "Indexes: fulltext 'chunk_text_ft' on Chunk.text, fulltext 'page_title_ft' on Page.title+summary."
)

SYSTEM_PROMPT = """You are a Vietnamese knowledge graph QA agent. You answer questions by querying a Neo4j graph database.

Available tools:
- kg_schema(): Returns the graph schema (node labels, relationships, indexes)
- kg_query(cypher): Execute a read-only Cypher query. Returns up to 10 rows.
- text_search(query): Fulltext search over text chunks. Returns top 5 results.
- get_passage(chunk_id): Get full text of a specific chunk by ID.
- entity_neighborhood(entity_name, hops=1): Find an entity and explore its neighborhood. Returns the entity, chunks mentioning it, and co-mentioned entities. Use hops=2 to expand to second-degree connections.
- path_search(entity_a, entity_b, max_hops=3): Find the shortest path between two entities in the graph. Useful for multi-hop questions connecting two concepts.

You MUST respond with a JSON object in one of these formats:

To use a tool:
{"thought": "your reasoning", "action": "tool_name", "action_input": {"param": "value"}}

To give final answer:
{"thought": "your reasoning", "final_answer": "your answer in Vietnamese with citations"}

Rules:
- Think step by step
- Use kg_schema() first if unsure about the graph structure
- For kg_query, write valid read-only Cypher (no CREATE/MERGE/DELETE/SET)
- Only use these schema labels/relationships verbatim: Page, Chunk, Entity, Person, Organization, Location, Work, Community, Question, HAS_CHUNK, MENTIONS, LINKS_TO, FOUNDED_BY, LOCATED_IN, BORN_IN, MEMBER_OF, PART_OF, CREATED_BY
- Only use these property keys verbatim: id, title, url, summary, text, sequence_number, embedding, name, type, source, aliases
- Do NOT invent Vietnamese schema terms like Người, Địa_điểm, ten, noi_sinh
- If unsure about Cypher, prefer text_search(query) over guessing schema
- Answer in Vietnamese
- Cite sources by mentioning page titles
- If you cannot find the answer after several attempts, say so honestly"""


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _tool_kg_schema(*, emit=None) -> str:
    if emit is not None:
        emit("tool_call", {"tool": "kg_schema", "input": {}})
    return _GRAPH_SCHEMA


def _assert_agent_readonly_cypher(cypher: str) -> None:
    """Allow exploratory read-only Cypher without forcing retrieval aliases."""
    raw = (cypher or "").strip()
    if not raw:
        raise RuntimeError("Generated Cypher is empty")

    trimmed = raw[:-1] if raw.endswith(";") else raw
    if ";" in trimmed:
        raise RuntimeError("Generated Cypher contains multiple statements")

    stripped = re.sub(r"//[^\n]*", " ", raw)
    stripped = re.sub(r"/\*.*?\*/", " ", stripped, flags=re.DOTALL)
    lowered = re.sub(r"\s+", " ", stripped.lower())

    blocked_keywords = [
        "create", "merge", "delete", "detach", "set",
        "remove", "drop", "load csv", "apoc.periodic", "call dbms",
    ]
    for kw in blocked_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            raise RuntimeError("Generated Cypher is not read-only")

    schema_identifiers = re.findall(r":\s*([A-Za-z_\u00C0-\u024F][A-Za-z0-9_\u00C0-\u024F]*)", raw)
    for ident in schema_identifiers:
        if ident not in _ALLOWED_SCHEMA_IDENTIFIERS:
            raise RuntimeError(f"Generated Cypher uses unknown schema identifier: {ident}")

    property_keys = set(re.findall(r"\.\s*([A-Za-z_\u00C0-\u024F][A-Za-z0-9_\u00C0-\u024F]*)", raw))
    property_keys.update(re.findall(r"\{\s*([A-Za-z_\u00C0-\u024F][A-Za-z0-9_\u00C0-\u024F]*)\s*:", raw))
    for key in property_keys:
        if key not in _ALLOWED_PROPERTY_KEYS:
            raise RuntimeError(f"Generated Cypher uses unknown schema property: {key}")


def _tool_kg_query(cypher: str, *, emit=None) -> str:
    if emit is not None:
        emit("tool_call", {"tool": "kg_query", "input": {"cypher": cypher}})
        emit("cypher", {"query": cypher})
    try:
        _assert_agent_readonly_cypher(cypher)
    except RuntimeError as e:
        if emit is not None:
            emit("tool_result", {"tool": "kg_query", "status": "error", "message": str(e)})
        return f"Error: Cypher validation failed - {e}"

    try:
        with neo4j_client.session() as session:
            records = session.run(cypher, top_k=10)
            rows = [dict(r) for r in records][:10]
        if not rows:
            if emit is not None:
                emit("tool_result", {"tool": "kg_query", "status": "empty", "row_count": 0})
            return "No results found."
        if emit is not None:
            emit("tool_result", {"tool": "kg_query", "status": "ok", "row_count": len(rows)})
        return json.dumps(rows, ensure_ascii=False, default=str)
    except Exception as e:
        if emit is not None:
            emit("tool_result", {"tool": "kg_query", "status": "error", "message": str(e)})
        return f"Error executing query: {e}"


def _tool_text_search(query: str, *, emit=None) -> str:
    if emit is not None:
        emit("tool_call", {"tool": "text_search", "input": {"query": query}})
    try:
        with neo4j_client.session() as session:
            records = session.run(
                """
                CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node, score
                MATCH (p:Page)-[:HAS_CHUNK]->(node)
                RETURN p.title AS page_title, p.url AS page_url,
                       node.id AS chunk_id, node.text AS chunk_text, score
                ORDER BY score DESC
                LIMIT 5
                """,
                q=query,
            )
            rows = [dict(r) for r in records]
        if not rows:
            if emit is not None:
                emit("tool_result", {"tool": "text_search", "status": "empty", "row_count": 0})
            return "No results found."
        if emit is not None:
            emit("tool_result", {"tool": "text_search", "status": "ok", "row_count": len(rows)})
        return json.dumps(rows, ensure_ascii=False, default=str)
    except Exception as e:
        if emit is not None:
            emit("tool_result", {"tool": "text_search", "status": "error", "message": str(e)})
        return f"Error in text search: {e}"


def _tool_get_passage(chunk_id: str, *, emit=None) -> str:
    if emit is not None:
        emit("tool_call", {"tool": "get_passage", "input": {"chunk_id": chunk_id}})
    try:
        with neo4j_client.session() as session:
            records = session.run(
                """
                MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk {id: $chunk_id})
                RETURN p.title AS page_title, p.url AS page_url, c.text AS chunk_text
                """,
                chunk_id=chunk_id,
            )
            rows = [dict(r) for r in records]
        if not rows:
            if emit is not None:
                emit("tool_result", {"tool": "get_passage", "status": "empty"})
            return "Chunk not found."
        if emit is not None:
            emit("tool_result", {"tool": "get_passage", "status": "ok", "row_count": 1})
        return json.dumps(rows[0], ensure_ascii=False, default=str)
    except Exception as e:
        if emit is not None:
            emit("tool_result", {"tool": "get_passage", "status": "error", "message": str(e)})
        return f"Error getting passage: {e}"


def _tool_entity_neighborhood(entity_name: str, hops: int = 1, *, emit=None) -> str:
    """Find an entity and its neighborhood (co-mentioned entities and chunks)."""
    hops = max(1, min(hops, 3))  # Clamp to 1-3
    if emit is not None:
        emit("tool_call", {"tool": "entity_neighborhood", "input": {"entity_name": entity_name, "hops": hops}})

    try:
        with neo4j_client.session() as session:
            if hops == 1:
                records = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($name)
                    WITH e LIMIT 1
                    OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
                    OPTIONAL MATCH (p:Page)-[:HAS_CHUNK]->(c)
                    OPTIONAL MATCH (c)-[:MENTIONS]->(co_entity:Entity)
                    WHERE co_entity <> e
                    RETURN e.name AS entity_name, e.type AS entity_type,
                           collect(DISTINCT {
                               chunk_id: c.id,
                               page_title: p.title,
                               chunk_text: left(c.text, 200)
                           })[..10] AS chunks,
                           collect(DISTINCT {name: co_entity.name, type: co_entity.type})[..10] AS co_entities
                    """,
                    name=entity_name,
                )
            else:
                records = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($name)
                    WITH e LIMIT 1
                    OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
                    OPTIONAL MATCH (p:Page)-[:HAS_CHUNK]->(c)
                    OPTIONAL MATCH (c)-[:MENTIONS]->(co1:Entity)
                    WHERE co1 <> e
                    WITH e, c, p, co1
                    OPTIONAL MATCH (c2:Chunk)-[:MENTIONS]->(co1)
                    WHERE c2 <> c
                    OPTIONAL MATCH (p2:Page)-[:HAS_CHUNK]->(c2)
                    OPTIONAL MATCH (c2)-[:MENTIONS]->(co2:Entity)
                    WHERE co2 <> e AND co2 <> co1
                    RETURN e.name AS entity_name, e.type AS entity_type,
                           collect(DISTINCT {
                               chunk_id: c.id,
                               page_title: p.title,
                               chunk_text: left(c.text, 200)
                           })[..10] AS chunks,
                           collect(DISTINCT {name: co1.name, type: co1.type})[..5] AS co_entities_hop1,
                           collect(DISTINCT {
                               chunk_id: c2.id,
                               page_title: p2.title,
                               chunk_text: left(c2.text, 200)
                           })[..5] AS chunks_hop2,
                           collect(DISTINCT {name: co2.name, type: co2.type})[..5] AS co_entities_hop2
                    """,
                    name=entity_name,
                )

            rows = [dict(r) for r in records]

        if not rows or rows[0].get("entity_name") is None:
            if emit is not None:
                emit("tool_result", {"tool": "entity_neighborhood", "status": "empty"})
            return f"Entity not found matching '{entity_name}'."

        row = rows[0]
        result: dict = {
            "entity": {"name": row["entity_name"], "type": row.get("entity_type")},
            "chunks": [ch for ch in row.get("chunks", []) if ch.get("chunk_id")],
        }

        if hops == 1:
            result["co_entities"] = [
                e for e in row.get("co_entities", []) if e.get("name")
            ]
        else:
            result["co_entities_hop1"] = [
                e for e in row.get("co_entities_hop1", []) if e.get("name")
            ]
            result["chunks_hop2"] = [
                ch for ch in row.get("chunks_hop2", []) if ch.get("chunk_id")
            ]
            result["co_entities_hop2"] = [
                e for e in row.get("co_entities_hop2", []) if e.get("name")
            ]

        if emit is not None:
            emit("tool_result", {"tool": "entity_neighborhood", "status": "ok", "row_count": len(result.get("chunks", []))})
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        if emit is not None:
            emit("tool_result", {"tool": "entity_neighborhood", "status": "error", "message": str(e)})
        return f"Error in entity neighborhood search: {e}"


def _tool_path_search(entity_a: str, entity_b: str, max_hops: int = 3, *, emit=None) -> str:
    """Find shortest path between two entities through the graph."""
    max_hops = max(1, min(max_hops, 5))  # Clamp to 1-5
    max_rels = max_hops * 2
    if emit is not None:
        emit(
            "tool_call",
            {"tool": "path_search", "input": {"entity_a": entity_a, "entity_b": entity_b, "max_hops": max_hops}},
        )

    try:
        with neo4j_client.session() as session:
            records = session.run(
                """
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
                        WHEN 'Entity' IN labels(n) THEN {label: 'Entity', name: n.name, type: n.type}
                        WHEN 'Chunk' IN labels(n) THEN {label: 'Chunk', id: n.id, text: left(n.text, 100)}
                        WHEN 'Page' IN labels(n) THEN {label: 'Page', title: n.title, url: n.url}
                        ELSE {label: head(labels(n)), id: n.id}
                    END
                ] AS path_nodes,
                [r IN relationships(path) | type(r)] AS rel_types,
                length(path) AS path_length
                """.replace("{max_rels}", str(max_rels)),
                name_a=entity_a,
                name_b=entity_b,
            )
            rows = [dict(r) for r in records]

        if not rows:
            if emit is not None:
                emit("tool_result", {"tool": "path_search", "status": "empty"})
            return f"No path found between '{entity_a}' and '{entity_b}'."

        row = rows[0]
        path_nodes = row.get("path_nodes", [])
        rel_types = row.get("rel_types", [])

        # Format path as readable string
        path_parts: list[str] = []
        for i, node in enumerate(path_nodes):
            label = node.get("label", "?")
            if label == "Entity":
                path_parts.append(f"[Entity: {node.get('name', '?')} ({node.get('type', '?')})]")
            elif label == "Chunk":
                path_parts.append(f"[Chunk: {node.get('id', '?')}]")
            elif label == "Page":
                path_parts.append(f"[Page: {node.get('title', '?')}]")
            else:
                path_parts.append(f"[{label}: {node.get('id', '?')}]")

            if i < len(rel_types):
                path_parts.append(f" -[:{rel_types[i]}]-> ")

        result = {
            "path": "".join(path_parts),
            "path_length": row.get("path_length"),
            "nodes": path_nodes,
            "relationships": rel_types,
        }
        if emit is not None:
            emit("tool_result", {"tool": "path_search", "status": "ok", "row_count": 1})
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        if emit is not None:
            emit("tool_result", {"tool": "path_search", "status": "error", "message": str(e)})
        return f"Error in path search: {e}"


def _call_tool(action: str, action_input: dict, emit=None) -> str:
    if action == "kg_schema":
        return _tool_kg_schema(emit=emit)
    if action == "kg_query":
        return _tool_kg_query(action_input.get("cypher", ""), emit=emit)
    if action == "text_search":
        return _tool_text_search(action_input.get("query", ""), emit=emit)
    if action == "get_passage":
        return _tool_get_passage(action_input.get("chunk_id", ""), emit=emit)
    if action == "entity_neighborhood":
        return _tool_entity_neighborhood(
            action_input.get("entity_name", ""),
            action_input.get("hops", 1),
            emit=emit,
        )
    if action == "path_search":
        return _tool_path_search(
            action_input.get("entity_a", ""),
            action_input.get("entity_b", ""),
            action_input.get("max_hops", 3),
            emit=emit,
        )
    return f"Error: Unknown tool '{action}'. Available: kg_schema, kg_query, text_search, get_passage, entity_neighborhood, path_search"


TOOLS = {
    "kg_schema": lambda inp: _call_tool("kg_schema", inp or {}, emit=None),
    "kg_query": lambda inp: _call_tool("kg_query", inp or {}, emit=None),
    "text_search": lambda inp: _call_tool("text_search", inp or {}, emit=None),
    "get_passage": lambda inp: _call_tool("get_passage", inp or {}, emit=None),
    "entity_neighborhood": lambda inp: _call_tool("entity_neighborhood", inp or {}, emit=None),
    "path_search": lambda inp: _call_tool("path_search", inp or {}, emit=None),
}


# ---------------------------------------------------------------------------
# Sufficiency gating
# ---------------------------------------------------------------------------


def _check_sufficiency(observations: list[str], question: str) -> tuple[bool, float]:
    """Check if collected evidence is sufficient to answer the question.

    Returns (is_sufficient, confidence_score).
    """
    valid_obs = [
        o for o in observations if o and not o.startswith("Error") and o != "No results found."
    ]

    if not valid_obs:
        return False, 0.0

    # Count unique chunks found
    chunk_ids: set[str] = set()
    for obs in valid_obs:
        try:
            data = json.loads(obs)
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict) and "chunk_id" in row:
                        chunk_ids.add(row["chunk_id"])
            elif isinstance(data, dict) and "chunk_id" in data:
                chunk_ids.add(data["chunk_id"])
        except (json.JSONDecodeError, TypeError):
            pass

    # Heuristic confidence
    confidence = min(1.0, len(chunk_ids) / 3.0)  # 3+ chunks = high confidence
    confidence += min(0.3, len(valid_obs) * 0.1)  # bonus for more observations
    confidence = min(1.0, confidence)

    return confidence >= 0.5, confidence


def _synthesize_from_observations(observations: list[str], citations: list[dict]) -> QueryResult:
    """Build a fallback answer from collected observations when agent doesn't converge."""
    has_valid_observation = any(
        obs and not obs.startswith("Error") and obs != "No results found."
        for obs in observations
    )

    if has_valid_observation:
        answer = "Không đủ thông tin trong cơ sở tri thức để trả lời câu hỏi này một cách chính xác."
    else:
        answer = "Không tìm thấy thông tin phù hợp trong cơ sở tri thức."

    trace: QueryTrace = {
        "tier": "generated",
        "steps": [
            {
                "kind": "answer",
                "name": "fallback.synthesized_observations",
                "status": "fallback",
            }
        ],
    }
    return QueryResult(answer=answer, citations=citations, trace=trace)


def _append_trace_step(
    trace_steps: list[QueryTraceStep],
    *,
    kind: Literal["retrieval", "ranking", "answer"],
    name: str,
    status: Literal["ok", "fallback", "empty"],
    row_count: int | None = None,
    top_k: int | None = None,
    error_type: str | None = None,
) -> None:
    step: QueryTraceStep = {"kind": kind, "name": name, "status": status}
    if row_count is not None:
        step["row_count"] = row_count
    if top_k is not None:
        step["top_k"] = top_k
    if error_type:
        step["error_type"] = error_type
    trace_steps.append(step)


# ---------------------------------------------------------------------------
# Core ReAct loop
# ---------------------------------------------------------------------------


def _extract_citations(observation: str, seen_chunk_ids: set[str], citations: list[dict]) -> list[dict]:
    """Extract chunk citations from a tool observation in-place."""
    new_citations: list[dict] = []
    try:
        obs_data = json.loads(observation) if not observation.startswith("Error") else None
    except (json.JSONDecodeError, TypeError):
        obs_data = None

    if obs_data:
        if isinstance(obs_data, list):
            for row in obs_data:
                if isinstance(row, dict) and "chunk_id" in row:
                    cid = row["chunk_id"]
                    if cid not in seen_chunk_ids:
                        seen_chunk_ids.add(cid)
                        citation = {
                            "page_title": row.get("page_title", ""),
                            "page_url": row.get("page_url", ""),
                            "chunk_id": cid,
                        }
                        citations.append(citation)
                        new_citations.append(citation)
        elif isinstance(obs_data, dict) and "chunk_id" in obs_data:
            cid = obs_data["chunk_id"]
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                citation = {
                    "page_title": obs_data.get("page_title", ""),
                    "page_url": obs_data.get("page_url", ""),
                    "chunk_id": cid,
                }
                citations.append(citation)
                new_citations.append(citation)
    return new_citations


def _recovery_hint(action: str, observation: str) -> str | None:
    lowered = (observation or "").lower()
    if action == "kg_query" and "unknown schema" in lowered:
        return (
            "Cypher vừa dùng sai schema của graph. Hãy gọi kg_schema() để xem đúng label/property, "
            "hoặc dùng text_search(query) nếu chưa chắc cách viết Cypher."
        )
    if action == "kg_query" and observation.startswith("Error:"):
        return "kg_query bị lỗi. Hãy dùng kg_schema() hoặc text_search(query) thay vì đoán Cypher."
    if action == "kg_query" and observation == "No results found.":
        return "kg_query không tìm thấy kết quả. Hãy thử text_search(query) để lấy bằng chứng trước."
    return None


def _extract_subject_from_question(question: str) -> str:
    text = (question or "").strip()
    text = re.sub(r"[?!.]+$", "", text).strip()
    text = re.sub(r"^(ai|cái gì|điều gì|what|who)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+là ai$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+là gì$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _exact_subject_rows(question: str, top_k: int) -> list[dict]:
    subject = _extract_subject_from_question(question)
    if len(subject) < 2:
        return []

    with neo4j_client.session() as session:
        records = session.run(
            """
            MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk)
            WHERE toLower(p.title) = toLower($subject)
            RETURN p.title AS page_title,
                   p.url AS page_url,
                   p.id AS page_id,
                   c.id AS chunk_id,
                   c.text AS chunk_text,
                   c.sequence_number AS sequence_number,
                   10.0 AS score
            ORDER BY c.sequence_number ASC
            LIMIT $top_k
            """,
            subject=subject,
            top_k=top_k,
        )
        return [dict(r) for r in records]


_QUESTION_STOPWORDS = {
    "ai", "cái", "gì", "nào", "như", "thế", "ra", "sao", "là", "có", "được",
    "về", "của", "vào", "trong", "trên", "ở", "khi", "sau", "đã", "theo",
    "bao", "nhiêu", "một", "những", "các", "điều", "việc", "này", "kia",
}

_OUTCOME_LEADS = ("nhưng", "tuy nhiên", "từ đây", "sau đó", "do đó", "vì vậy")


def _question_keywords(question: str) -> set[str]:
    tokens = re.findall(r"\w+", (question or "").lower(), flags=re.UNICODE)
    return {
        token
        for token in tokens
        if len(token) >= 2 and token not in _QUESTION_STOPWORDS
    }


def _sentence_candidates(rows: list[dict]) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = []
    for row_index, row in enumerate(rows):
        text = str((row.get("chunk_text") or "")).strip().replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            normalized = sentence.strip(" ,;")
            if normalized:
                candidates.append((normalized, row_index))
    return candidates


def _clean_sentence_answer(sentence: str) -> str:
    cleaned = (sentence or "").strip()
    cleaned = re.sub(
        r"^(nhưng|tuy nhiên|từ đây|sau đó|do đó|vì vậy)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _select_descriptive_phrase(question: str, rows: list[dict]) -> str | None:
    question_keywords = _question_keywords(question)
    best_phrase: str | None = None
    best_score = float("-inf")

    for sentence, _row_index in _sentence_candidates(rows):
        match = re.search(
            r"\b(rất|vô cùng|cực kỳ|khá)\s+[^,.;]{3,120}",
            sentence,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        phrase = match.group(0).strip(" ,;")
        sentence_tokens = set(re.findall(r"\w+", sentence.lower(), flags=re.UNICODE))
        overlap = len(question_keywords & sentence_tokens)
        phrase_len = len(phrase.split())
        score = overlap * 3 + phrase_len * 2
        if " và " in phrase.lower():
            score += 4
        if score > best_score:
            best_score = score
            best_phrase = phrase

    return best_phrase


def _select_relevant_sentence(question: str, rows: list[dict]) -> str | None:
    question_keywords = _question_keywords(question)
    candidates = _sentence_candidates(rows)
    lowered_question = (question or "").lower()

    if "sau khi" in lowered_question:
        for index in range(1, len(candidates)):
            prev_sentence = candidates[index - 1][0]
            sentence = candidates[index][0]
            lowered = sentence.lower()
            if not any(lowered.startswith(prefix) for prefix in _OUTCOME_LEADS):
                continue
            prev_tokens = set(re.findall(r"\w+", prev_sentence.lower(), flags=re.UNICODE))
            prev_overlap = len(question_keywords & prev_tokens)
            if prev_overlap >= 4:
                return sentence

    best_sentence: str | None = None
    best_score = float("-inf")

    for index, (sentence, _row_index) in enumerate(candidates):
        sentence_tokens = set(re.findall(r"\w+", sentence.lower(), flags=re.UNICODE))
        overlap = len(question_keywords & sentence_tokens)
        score = overlap * 10 + min(len(sentence), 240) / 100.0

        lowered = sentence.lower()
        if any(lowered.startswith(prefix) for prefix in _OUTCOME_LEADS):
            score += 6
            if index > 0:
                prev_sentence = candidates[index - 1][0]
                prev_tokens = set(re.findall(r"\w+", prev_sentence.lower(), flags=re.UNICODE))
                prev_overlap = len(question_keywords & prev_tokens)
                if prev_overlap >= 3:
                    score += 12

        if re.search(r"\b(rất|vô cùng|cực kỳ|khá)\b", lowered):
            score += 3

        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence


def _compose_fallback_answer(question: str, rows: list[dict]) -> str:
    subject = _extract_subject_from_question(question)
    if subject:
        pattern = re.compile(rf"^{re.escape(subject)}(?:\s*\([^)]*\))?(?P<rest>.*)$", flags=re.IGNORECASE)
        for row in rows:
            text = str((row.get("chunk_text") or "")).strip().replace("\n", " ")
            text = re.sub(r"\s+", " ", text)
            match = pattern.match(text)
            if not match:
                continue

            remainder = match.group("rest").strip(" ,")
            direct_match = re.match(r"^là\s+(.+)$", remainder, flags=re.IGNORECASE)
            comma_match = re.search(r",\s+là\s+(.+)$", remainder, flags=re.IGNORECASE)
            predicate = None
            if direct_match:
                predicate = direct_match.group(1)
            elif comma_match:
                predicate = comma_match.group(1)

            if predicate:
                predicate = predicate.split(".")[0].strip(" ,;")
                if predicate:
                    return f"{subject} là {predicate}."

    if any(token in question.lower() for token in ("như thế nào", "ra sao", "tính chất")):
        phrase = _select_descriptive_phrase(question, rows)
        if phrase:
            return phrase
        sentence = _select_relevant_sentence(question, rows)
        if sentence:
            cleaned = _clean_sentence_answer(sentence)
            return cleaned if cleaned.endswith((".", "!", "?")) else cleaned + "."

    snippets = []
    for row in rows:
        chunk_text = str((row.get("chunk_text") or "")).strip().replace("\n", " ")
        if chunk_text:
            snippets.append(chunk_text[:220])
        if len(snippets) == 2:
            break

    if snippets:
        return "Không đủ thông tin trong cơ sở tri thức để trả lời câu hỏi này một cách chính xác."
    return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức."


def _expand_same_page_chunks(rows: list[dict], limit_per_page: int = 4) -> list[dict]:
    page_ids = []
    seen_page_ids: set[str] = set()
    for row in rows:
        page_id = row.get("page_id")
        if not page_id or page_id in seen_page_ids:
            continue
        seen_page_ids.add(page_id)
        page_ids.append(page_id)
        if len(page_ids) == 3:
            break

    if not page_ids:
        return []

    with neo4j_client.session() as session:
        records = session.run(
            """
            MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk)
            WHERE p.id IN $page_ids
            RETURN p.title AS page_title,
                   p.url AS page_url,
                   p.id AS page_id,
                   c.id AS chunk_id,
                   c.text AS chunk_text,
                   c.sequence_number AS sequence_number,
                   0.0 AS score
            ORDER BY p.id ASC, c.sequence_number ASC
            LIMIT $limit
            """,
            page_ids=page_ids,
            limit=max(1, len(page_ids) * limit_per_page),
        )
        return [dict(r) for r in records]


def _deterministic_retrieval_fallback(
    question: str,
    top_k: int,
    seen_chunk_ids: set[str],
    citations: list[dict],
    trace_steps: list[QueryTraceStep] | None = None,
) -> QueryResult | None:
    """Use deterministic retrieval when the agent stops producing valid actions."""
    local_trace_steps = trace_steps if trace_steps is not None else []

    try:
        rows = _exact_subject_rows(question, top_k=max(top_k, 4))
        _append_trace_step(
            local_trace_steps,
            kind="retrieval",
            name="retrieval.exact_subject",
            status="ok" if rows else "empty",
            row_count=len(rows),
            top_k=max(top_k, 4),
        )
    except Exception as exc:
        logger.warning("Exact-subject fallback retrieval failed", extra={"error": str(exc)})
        rows = []
        _append_trace_step(
            local_trace_steps,
            kind="retrieval",
            name="retrieval.exact_subject",
            status="fallback",
            error_type=type(exc).__name__,
            top_k=max(top_k, 4),
        )
    if not rows:
        try:
            rows = hybrid_retrieve(question, top_k=max(top_k * 3, 8))
            _append_trace_step(
                local_trace_steps,
                kind="retrieval",
                name="retrieval.hybrid",
                status="ok" if rows else "empty",
                row_count=len(rows),
                top_k=max(top_k * 3, 8),
            )
        except Exception as exc:
            logger.warning("Hybrid fallback retrieval failed", extra={"error": str(exc)})
            rows = []
            _append_trace_step(
                local_trace_steps,
                kind="retrieval",
                name="retrieval.hybrid",
                status="fallback",
                error_type=type(exc).__name__,
                top_k=max(top_k * 3, 8),
            )

    if not rows:
        observation = _tool_text_search(question)
        if observation.startswith("Error") or observation == "No results found.":
            _append_trace_step(
                local_trace_steps,
                kind="retrieval",
                name="tool.text_search",
                status="empty" if observation == "No results found." else "fallback",
                error_type="tool_error" if observation.startswith("Error") else None,
            )
            return None
        rows = json.loads(observation)
        _append_trace_step(
            local_trace_steps,
            kind="retrieval",
            name="tool.text_search",
            status="fallback",
            row_count=len(rows) if isinstance(rows, list) else 0,
            top_k=5,
        )
    else:
        try:
            expanded_rows = _expand_same_page_chunks(rows)
            _append_trace_step(
                local_trace_steps,
                kind="retrieval",
                name="retrieval.same_page_expansion",
                status="ok" if expanded_rows else "empty",
                row_count=len(expanded_rows),
            )
        except Exception as exc:
            logger.warning("Same-page expansion failed", extra={"error": str(exc)})
            expanded_rows = []
            _append_trace_step(
                local_trace_steps,
                kind="retrieval",
                name="retrieval.same_page_expansion",
                status="fallback",
                error_type=type(exc).__name__,
            )
        combined_rows: list[dict] = []
        seen_chunk_map: set[str] = set()
        for row in rows + expanded_rows:
            chunk_id = row.get("chunk_id")
            if not chunk_id or chunk_id in seen_chunk_map:
                continue
            seen_chunk_map.add(chunk_id)
            combined_rows.append(row)
        try:
            rows = rerank(question, combined_rows, text_key="chunk_text", top_k=max(top_k * 3, 8))
            _append_trace_step(
                local_trace_steps,
                kind="ranking",
                name="ranking.rerank",
                status="ok" if rows else "empty",
                row_count=len(rows),
                top_k=max(top_k * 3, 8),
            )
        except Exception as exc:
            logger.warning("Fallback rerank failed", extra={"error": str(exc)})
            rows = combined_rows[: max(top_k * 3, 8)]
            _append_trace_step(
                local_trace_steps,
                kind="ranking",
                name="ranking.rerank",
                status="fallback",
                error_type=type(exc).__name__,
                row_count=len(rows),
                top_k=max(top_k * 3, 8),
            )

    final_rows = rows[:top_k]
    observation = json.dumps(final_rows, ensure_ascii=False, default=str)
    _extract_citations(observation, seen_chunk_ids, citations)
    answer = _compose_fallback_answer(question, rows if isinstance(rows, list) else [])
    _append_trace_step(
        local_trace_steps,
        kind="answer",
        name="fallback.deterministic_retrieval",
        status="fallback",
        row_count=len(final_rows),
        top_k=top_k,
    )
    trace: QueryTrace = {"tier": "generated", "steps": local_trace_steps}
    return QueryResult(answer=answer, citations=citations, retrieval_tier="hybrid", trace=trace)


def agent_query_standard(question: str, top_k: int = 4, emit=None) -> QueryResult:
    """Standard ReAct agent loop without decomposition."""
    from src.infrastructure.local_llm import chat

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    citations: list[dict] = []
    observations: list[str] = []
    actions_taken: list[str] = []
    seen_chunk_ids: set[str] = set()
    consecutive_unparseable = 0
    trace_steps: list[QueryTraceStep] = []
    _append_trace_step(trace_steps, kind="retrieval", name="route.local_agent", status="ok")
    if emit is not None:
        emit("route", {"route": "local_agent"})

    for iteration in range(MAX_ITERATIONS):
        llm_attempts = 0
        raw = None
        while llm_attempts < 2:
            try:
                raw = chat(messages, max_new_tokens=512, temperature=0.1)
                break
            except Exception as e:
                llm_attempts += 1
                if llm_attempts >= 2:
                    logger.warning("Agent LLM call failed after retry", extra={"iteration": iteration, "error": str(e)})
                else:
                    logger.debug("Agent LLM call failed, retrying", extra={"iteration": iteration, "error": str(e)})

        if raw is None:
            break

        parsed = parse_agent_response(raw)
        if not parsed:
            logger.warning("Agent produced unparseable output", extra={"iteration": iteration, "raw": raw[:200]})
            consecutive_unparseable += 1
            if consecutive_unparseable >= 2:
                fallback_result = _deterministic_retrieval_fallback(
                    question, top_k, seen_chunk_ids, citations, trace_steps
                )
                if fallback_result is not None:
                    if emit is not None:
                        emit("fallback", {"name": "deterministic_retrieval"})
                        logger.info(
                            "Agent fell back to deterministic retrieval after unparseable output",
                            extra={"iteration": iteration, "consecutive_unparseable": consecutive_unparseable},
                    )
                    return fallback_result
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Please respond with valid JSON."})
            continue
        consecutive_unparseable = 0

        if "final_answer" in parsed:
            if not actions_taken:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "You must use at least one tool before providing a final answer. Use kg_query or text_search to find evidence first."})
                continue
            answer = parsed["final_answer"]
            logger.info("Agent converged", extra={"iterations": iteration + 1})
            _append_trace_step(trace_steps, kind="answer", name="answer.agent_final", status="ok")
            if emit is not None:
                emit("answer_delta", {"text": answer})
            trace: QueryTrace = {"tier": "generated", "steps": trace_steps}
            return QueryResult(answer=answer, citations=citations, trace=trace)

        action = parsed.get("action", "")
        action_input = parsed.get("action_input", {})
        if isinstance(action_input, str):
            action_input = {"query": action_input} if action == "text_search" else {"cypher": action_input}

        observation = _call_tool(action, action_input, emit=emit)

        observations.append(observation)
        actions_taken.append(action)
        _append_trace_step(
            trace_steps,
            kind="retrieval",
            name=f"tool.{action}" if action else "tool.unknown",
            status="empty" if observation == "No results found." else ("fallback" if observation.startswith("Error") else "ok"),
            error_type="tool_error" if observation.startswith("Error") else None,
        )

        new_citations = _extract_citations(observation, seen_chunk_ids, citations)
        if emit is not None:
            for citation in new_citations:
                emit("citation", citation)

        messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
        messages.append({"role": "user", "content": f"Observation: {observation[:2000]}"})
        hint = _recovery_hint(action, observation)
        if hint:
            messages.append({"role": "user", "content": hint})

        logger.debug("Agent iteration", extra={"iteration": iteration, "action": action})

        # Sufficiency gating: strategy switch at iteration 3
        if iteration == 2:
            is_sufficient, confidence = _check_sufficiency(observations, question)
            if not is_sufficient:
                prev_actions = set(actions_taken)
                if prev_actions <= {"kg_query"}:
                    hint = "Truy vấn Cypher không tìm đủ thông tin. Hãy thử text_search."
                elif prev_actions <= {"text_search"}:
                    hint = "Tìm kiếm văn bản không đủ. Hãy thử kg_query với Cypher."
                else:
                    hint = "Hãy thử cách tiếp cận khác để tìm thêm thông tin."
                logger.info(
                    "Sufficiency check failed at iteration 3, suggesting strategy switch",
                    extra={"confidence": confidence, "prev_actions": list(prev_actions)},
                )
                messages.append({"role": "user", "content": f"Observation: {hint}"})

        # Sufficiency gating: abstain at iteration 5
        if iteration == 4:
            is_sufficient, confidence = _check_sufficiency(observations, question)
            if not is_sufficient:
                logger.warning(
                    "Agent abstaining due to insufficient evidence",
                    extra={"confidence": confidence, "iterations": iteration + 1},
                )
                if emit is not None:
                    emit("fallback", {"name": "insufficient_evidence"})
                return QueryResult(
                    answer="Không đủ thông tin trong cơ sở tri thức để trả lời câu hỏi này một cách chính xác.",
                    citations=citations,
                    trace={
                        "tier": "generated",
                        "steps": trace_steps + [
                            {
                                "kind": "answer",
                                "name": "fallback.insufficient_evidence",
                                "status": "fallback",
                            }
                        ],
                    },
                )

    logger.warning("Agent did not converge", extra={"iterations": MAX_ITERATIONS})
    fallback_result = _deterministic_retrieval_fallback(question, top_k, seen_chunk_ids, citations, trace_steps)
    if fallback_result is not None:
        if emit is not None:
            emit("fallback", {"name": "deterministic_retrieval"})
            emit("answer_delta", {"text": fallback_result.answer})
        return fallback_result
    synthesized = _synthesize_from_observations(observations, citations)
    if synthesized.trace is not None:
        synthesized.trace["steps"] = trace_steps + synthesized.trace["steps"]
    if emit is not None:
        emit("fallback", {"name": "synthesized_observations"})
        emit("answer_delta", {"text": synthesized.answer})
    return synthesized


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def agent_query(question: str, top_k: int = 4, emit=None) -> QueryResult:
    """Run ReAct agent loop to answer a question using graph tools.

    For complex questions (complexity >= 3), decomposes into sub-questions first.
    """
    # Detect complexity and decompose if needed
    complexity = detect_complexity(question)

    if complexity >= COMPLEXITY_THRESHOLD:
        sub_questions = decompose_question(question)
        if sub_questions and emit is None:
            logger.info(f"Using decomposition for complex question (complexity={complexity})")
            return agent_query_with_decomposition(question, sub_questions)

    # Fall back to standard agent loop for simple questions
    return agent_query_standard(question, top_k, emit=emit)
