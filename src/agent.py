"""ReAct agent loop with graph tools for multi-hop question answering."""

from __future__ import annotations

import json
import re

from src.llm import assert_readonly_cypher
from src.logging_utils import get_logger
from src.neo4j_client import neo4j_client
from src.retrieve import QueryResult

logger = get_logger(__name__)

MAX_ITERATIONS = 6

_GRAPH_SCHEMA = (
    "Node labels: Page(id, title, url, summary), Chunk(id, text, sequence_number, embedding), "
    "Entity(id, name, type) with optional labels Person/Organization/Location/Work.\n"
    "Relationships: (Page)-[:HAS_CHUNK]->(Chunk), (Chunk)-[:MENTIONS]->(Entity), "
    "(Page)-[:LINKS_TO]->(Page).\n"
    "Indexes: fulltext 'chunk_text_ft' on Chunk.text, fulltext 'page_title_ft' on Page.title+summary."
)

_SYSTEM_PROMPT = """You are a Vietnamese knowledge graph QA agent. You answer questions by querying a Neo4j graph database.

Available tools:
- kg_schema(): Returns the graph schema (node labels, relationships, indexes)
- kg_query(cypher): Execute a read-only Cypher query. Returns up to 10 rows.
- text_search(query): Fulltext search over text chunks. Returns top 5 results.
- get_passage(chunk_id): Get full text of a specific chunk by ID.

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


_TOOLS = {
    "kg_schema": lambda _: _tool_kg_schema(),
    "kg_query": lambda inp: _tool_kg_query(inp.get("cypher", "")),
    "text_search": lambda inp: _tool_text_search(inp.get("query", "")),
    "get_passage": lambda inp: _tool_get_passage(inp.get("chunk_id", "")),
}


def _parse_agent_response(text: str) -> dict | None:
    """Parse JSON from model output, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _synthesize_from_observations(observations: list[str], citations: list) -> QueryResult:
    """Build a fallback answer from collected observations when agent doesn't converge."""
    snippets = []
    for obs in observations:
        if obs and not obs.startswith("Error") and obs != "No results found.":
            snippets.append(obs[:300])

    if snippets:
        answer = "Dựa trên thông tin tìm được: " + " | ".join(snippets[:3])
    else:
        answer = "Không tìm thấy thông tin phù hợp trong cơ sở tri thức."

    return QueryResult(answer=answer, citations=citations, strategy="agent", strategy_used="agent")


def agent_query(question: str, top_k: int = 4) -> QueryResult:
    """Run ReAct agent loop to answer a question using graph tools."""
    from src.local_llm import chat

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    citations: list = []
    observations: list[str] = []
    seen_chunk_ids: set[str] = set()

    for iteration in range(MAX_ITERATIONS):
        try:
            raw = chat(messages, max_new_tokens=512, temperature=0.1)
        except Exception as e:
            logger.warning("Agent LLM call failed", extra={"iteration": iteration, "error": str(e)})
            break

        parsed = _parse_agent_response(raw)
        if not parsed:
            logger.warning("Agent produced unparseable output", extra={"iteration": iteration, "raw": raw[:200]})
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Please respond with valid JSON."})
            continue

        if "final_answer" in parsed:
            answer = parsed["final_answer"]
            logger.info("Agent converged", extra={"iterations": iteration + 1})
            return QueryResult(answer=answer, citations=citations, strategy="agent", strategy_used="agent")

        action = parsed.get("action", "")
        action_input = parsed.get("action_input", {})
        if isinstance(action_input, str):
            action_input = {"query": action_input} if action == "text_search" else {"cypher": action_input}

        tool_fn = _TOOLS.get(action)
        if not tool_fn:
            observation = f"Error: Unknown tool '{action}'. Available: kg_schema, kg_query, text_search, get_passage"
        else:
            observation = tool_fn(action_input)

        observations.append(observation)

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

        messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
        messages.append({"role": "user", "content": f"Observation: {observation[:2000]}"})

        logger.debug("Agent iteration", extra={"iteration": iteration, "action": action})

    logger.warning("Agent did not converge", extra={"iterations": MAX_ITERATIONS})
    return _synthesize_from_observations(observations, citations)
