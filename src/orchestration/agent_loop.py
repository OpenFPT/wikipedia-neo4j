"""Core ReAct agent loop: think → act → observe."""

from __future__ import annotations

import json

from src.infrastructure.llm import assert_readonly_cypher
from src.logging_utils import get_logger
from src.infrastructure.neo4j_client import neo4j_client
from src.retrieval.fusion import QueryResult

from src.orchestration._parsing import parse_agent_response
from src.orchestration.complexity import detect_complexity, COMPLEXITY_THRESHOLD
from src.orchestration.decomposition import decompose_question, agent_query_with_decomposition

logger = get_logger(__name__)

MAX_ITERATIONS = 6

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
- Answer in Vietnamese
- Cite sources by mentioning page titles
- If you cannot find the answer after several attempts, say so honestly"""


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _tool_kg_schema() -> str:
    return _GRAPH_SCHEMA


def _tool_kg_query(cypher: str) -> str:
    try:
        assert_readonly_cypher(cypher)
    except RuntimeError as e:
        return f"Error: Cypher validation failed - {e}"

    try:
        with neo4j_client.session() as session:
            records = session.run(cypher)
            rows = [dict(r) for r in records][:10]
        if not rows:
            return "No results found."
        return json.dumps(rows, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error executing query: {e}"


def _tool_text_search(query: str) -> str:
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
            return "No results found."
        return json.dumps(rows, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error in text search: {e}"


def _tool_get_passage(chunk_id: str) -> str:
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
            return "Chunk not found."
        return json.dumps(rows[0], ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error getting passage: {e}"


def _tool_entity_neighborhood(entity_name: str, hops: int = 1) -> str:
    """Find an entity and its neighborhood (co-mentioned entities and chunks)."""
    hops = max(1, min(hops, 3))  # Clamp to 1-3

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

        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error in entity neighborhood search: {e}"


def _tool_path_search(entity_a: str, entity_b: str, max_hops: int = 3) -> str:
    """Find shortest path between two entities through the graph."""
    max_hops = max(1, min(max_hops, 5))  # Clamp to 1-5
    max_rels = max_hops * 2

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
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error in path search: {e}"


TOOLS = {
    "kg_schema": lambda _: _tool_kg_schema(),
    "kg_query": lambda inp: _tool_kg_query(inp.get("cypher", "")),
    "text_search": lambda inp: _tool_text_search(inp.get("query", "")),
    "get_passage": lambda inp: _tool_get_passage(inp.get("chunk_id", "")),
    "entity_neighborhood": lambda inp: _tool_entity_neighborhood(
        inp.get("entity_name", ""), inp.get("hops", 1)
    ),
    "path_search": lambda inp: _tool_path_search(
        inp.get("entity_a", ""), inp.get("entity_b", ""), inp.get("max_hops", 3)
    ),
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
    snippets = []
    for obs in observations:
        if obs and not obs.startswith("Error") and obs != "No results found.":
            snippets.append(obs[:300])

    if snippets:
        answer = "Dựa trên thông tin tìm được: " + " | ".join(snippets[:3])
    else:
        answer = "Không tìm thấy thông tin phù hợp trong cơ sở tri thức."

    return QueryResult(answer=answer, citations=citations)


# ---------------------------------------------------------------------------
# Core ReAct loop
# ---------------------------------------------------------------------------


def _extract_citations(observation: str, seen_chunk_ids: set[str], citations: list[dict]) -> None:
    """Extract chunk citations from a tool observation in-place."""
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
                        citations.append({
                            "page_title": row.get("page_title", ""),
                            "page_url": row.get("page_url", ""),
                            "chunk_id": cid,
                        })
        elif isinstance(obs_data, dict) and "chunk_id" in obs_data:
            cid = obs_data["chunk_id"]
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                citations.append({
                    "page_title": obs_data.get("page_title", ""),
                    "page_url": obs_data.get("page_url", ""),
                    "chunk_id": cid,
                })


def agent_query_standard(question: str, top_k: int = 4) -> QueryResult:
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
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Please respond with valid JSON."})
            continue

        if "final_answer" in parsed:
            if not actions_taken:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "You must use at least one tool before providing a final answer. Use kg_query or text_search to find evidence first."})
                continue
            answer = parsed["final_answer"]
            logger.info("Agent converged", extra={"iterations": iteration + 1})
            return QueryResult(answer=answer, citations=citations)

        action = parsed.get("action", "")
        action_input = parsed.get("action_input", {})
        if isinstance(action_input, str):
            action_input = {"query": action_input} if action == "text_search" else {"cypher": action_input}

        tool_fn = TOOLS.get(action)
        if not tool_fn:
            observation = f"Error: Unknown tool '{action}'. Available: kg_schema, kg_query, text_search, get_passage, entity_neighborhood, path_search"
        else:
            observation = tool_fn(action_input)

        observations.append(observation)
        actions_taken.append(action)

        _extract_citations(observation, seen_chunk_ids, citations)

        messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
        messages.append({"role": "user", "content": f"Observation: {observation[:2000]}"})

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
                return QueryResult(
                    answer="Không đủ thông tin trong cơ sở tri thức để trả lời câu hỏi này một cách chính xác.",
                    citations=citations,
                )

    logger.warning("Agent did not converge", extra={"iterations": MAX_ITERATIONS})
    return _synthesize_from_observations(observations, citations)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def agent_query(question: str, top_k: int = 4) -> QueryResult:
    """Run ReAct agent loop to answer a question using graph tools.

    For complex questions (complexity >= 3), decomposes into sub-questions first.
    """
    # Detect complexity and decompose if needed
    complexity = detect_complexity(question)

    if complexity >= COMPLEXITY_THRESHOLD:
        sub_questions = decompose_question(question)
        if sub_questions:
            logger.info(f"Using decomposition for complex question (complexity={complexity})")
            return agent_query_with_decomposition(question, sub_questions)

    # Fall back to standard agent loop for simple questions
    return agent_query_standard(question, top_k)
