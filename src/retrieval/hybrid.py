"""Graph retrieval and deterministic answer assembly."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from src.config import settings
from src.infrastructure.llm import assert_readonly_cypher, embed_texts, generate_readonly_cypher, _client_pool
from src.logging_utils import get_logger
from neo4j.exceptions import CypherSyntaxError
from src.infrastructure.neo4j_client import neo4j_client
from src.retrieval.planner import plan as plan_query
from src.retrieval.templates import build_fulltext_query, pick_template


logger = get_logger(__name__)

_PAGE_INDEX_CACHE: dict[str, dict] | None = None


def _load_page_index() -> dict[str, dict]:
    """Build (and cache) a simple accent-insensitive Page title index."""
    global _PAGE_INDEX_CACHE
    if _PAGE_INDEX_CACHE is not None:
        return _PAGE_INDEX_CACHE

    idx: dict[str, dict] = {}
    with neo4j_client.session() as session:
        records = session.run("MATCH (p:Page) RETURN p.id AS id, p.title AS title, p.url AS url")
        for r in records:
            title = r.get("title") or ""
            key = _normalize_for_tokens(title)
            if not key:
                continue
            # Keep first; collisions are rare and not fatal.
            idx.setdefault(key, {"id": r.get("id"), "title": title, "url": r.get("url")})

    _PAGE_INDEX_CACHE = idx
    return idx


_Q_STOPWORDS = {
    "la",
    "ai",
    "gi",
    "cai",
    "cach",
    "nao",
    "o",
    "tai",
    "trong",
    "bao",
    "nhieu",
    "mot",
    "nhung",
    "cua",
    "co",
    "khong",
}


def _extract_subject_phrase(question: str) -> str:
    toks = _normalize_for_tokens(question).split()
    toks = [t for t in toks if t not in _Q_STOPWORDS]
    # Keep a short phrase; good for person/place/page names.
    return " ".join(toks[:6]).strip()


def _resolve_page_by_title(question: str) -> dict | None:
    """Try to resolve a Page by accent-insensitive title match."""
    subject = _extract_subject_phrase(question)
    if not subject:
        return None

    idx = _load_page_index()

    # Exact key match
    if subject in idx:
        return idx[subject]

    # Substring match (pick shortest title that contains the subject)
    matches = []
    for key, meta in idx.items():
        if subject and subject in key:
            matches.append((len(key), meta))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[0][1]


def _strip_accents(s: str) -> str:
    # NFKD + drop combining marks
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _normalize_for_tokens(s: str) -> str:
    s = _strip_accents(s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _key_terms(question: str) -> set[str]:
    toks = _normalize_for_tokens(question).split()
    return {t for t in toks if len(t) >= 4}


@dataclass
class QueryResult:
    """Query response model used by API layer."""

    answer: str
    citations: list[dict]
    retrieval_tier: str = "unknown"


_LEGACY_HYBRID_CYPHER = """
CALL {
  CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node, score
  MATCH (p:Page)-[:HAS_CHUNK]->(node)
  RETURN p.title AS page_title,
         p.url AS page_url,
         p.id AS page_id,
         node.id AS chunk_id,
         node.text AS chunk_text,
         score * 1.0 AS score
  LIMIT $top_k

  UNION

  CALL db.index.fulltext.queryNodes('page_title_ft', $q) YIELD node, score
  MATCH (node:Page)-[:HAS_CHUNK]->(c:Chunk)
  RETURN node.title AS page_title,
         node.url AS page_url,
         node.id AS page_id,
         c.id AS chunk_id,
         c.text AS chunk_text,
         score * 0.8 AS score
  LIMIT $top_k

  UNION

  CALL db.index.fulltext.queryNodes('entity_alias_ft', $q) YIELD node, score
  MATCH (c:Chunk)-[:MENTIONS]->(node)
  MATCH (p:Page)-[:HAS_CHUNK]->(c)
  RETURN p.title AS page_title,
         p.url AS page_url,
         p.id AS page_id,
         c.id AS chunk_id,
         c.text AS chunk_text,
         score * 0.7 AS score
  LIMIT $top_k
}
WITH page_title, page_url, page_id, chunk_id, chunk_text, max(score) AS score
RETURN page_title, page_url, page_id, chunk_id, chunk_text, score
ORDER BY score DESC
LIMIT $top_k
"""

_BM25_CYPHER = """
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node, score
MATCH (p:Page)-[:HAS_CHUNK]->(node)
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       node.id AS chunk_id,
       node.text AS chunk_text,
       score AS bm25_score
ORDER BY score DESC
LIMIT $top_k
"""

_TITLE_CYPHER = """
CALL db.index.fulltext.queryNodes('page_title_ft', $q) YIELD node, score
MATCH (node:Page)-[:HAS_CHUNK]->(c:Chunk)
RETURN node.title AS page_title,
       node.url AS page_url,
       node.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS title_score
ORDER BY score DESC
LIMIT $top_k
"""

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

_GRAPH_CYPHER = """
CALL db.index.fulltext.queryNodes('entity_alias_ft', $q) YIELD node AS e, score
MATCH (c:Chunk)-[:MENTIONS]->(e)
MATCH (p:Page)-[:HAS_CHUNK]->(c)
RETURN DISTINCT p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS graph_score
ORDER BY score DESC
LIMIT $top_k
"""

_EXPAND_LINKS_CYPHER = """
MATCH (source:Page)-[:LINKS_TO]->(linked:Page)-[:HAS_CHUNK]->(c:Chunk)
WHERE source.id IN $page_ids
WITH linked, c
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node, score
WHERE node = c
MATCH (linked)-[:HAS_CHUNK]->(node)
RETURN linked.title AS page_title,
       linked.url AS page_url,
       linked.id AS page_id,
       node.id AS chunk_id,
       node.text AS chunk_text,
       score * 0.9 AS score
ORDER BY score DESC
LIMIT $top_k
"""

_PAGE_CHUNKS_CYPHER = """
MATCH (p:Page {id: $page_id})-[:HAS_CHUNK]->(c:Chunk)
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       1.0 AS score
ORDER BY c.sequence_number ASC
LIMIT $top_k
"""


def _run_bm25_query(question: str, top_k: int) -> list[dict]:  # pragma: no cover
    """Execute BM25 fulltext search."""
    q1 = (question or "").strip()
    q2 = _strip_accents(q1)
    queries = [q1]
    if q2 and q2 != q1:
        queries.append(q2)

    rows: list[dict] = []
    with neo4j_client.session() as session:
        for q in queries:
            records = session.run(_BM25_CYPHER, q=q, top_k=top_k)
            rows.extend(dict(r) for r in records)

    # Dedup by chunk_id, keep max bm25_score
    best: dict[str, dict] = {}
    for r in rows:
        cid = r.get("chunk_id")
        if not cid:
            continue
        if cid not in best or (r.get("bm25_score", 0) > best[cid].get("bm25_score", 0)):
            best[cid] = r
    rows = list(best.values())
    rows.sort(key=lambda x: x.get("bm25_score", 0), reverse=True)
    rows = rows[:top_k]
    logger.info("BM25 retrieval executed", extra={"rows": len(rows)})
    return rows


def _run_title_query(question: str, top_k: int) -> list[dict]:  # pragma: no cover
    """Execute fulltext search over Page titles, returning their chunks."""
    q1 = (question or "").strip()
    q2 = _strip_accents(q1)
    queries = [q1]
    if q2 and q2 != q1:
        queries.append(q2)

    rows: list[dict] = []
    with neo4j_client.session() as session:
        for q in queries:
            records = session.run(_TITLE_CYPHER, q=q, top_k=top_k)
            rows.extend(dict(r) for r in records)

    best: dict[str, dict] = {}
    for r in rows:
        cid = r.get("chunk_id")
        if not cid:
            continue
        if cid not in best or (r.get("title_score", 0) > best[cid].get("title_score", 0)):
            best[cid] = r
    rows = list(best.values())
    rows.sort(key=lambda x: x.get("title_score", 0), reverse=True)
    return rows[:top_k]


def _run_vector_query(
    question: str, top_k: int, *, query_embedding: list[float] | None = None
) -> list[dict]:  # pragma: no cover
    """Execute vector similarity search."""
    if settings.neo4j_use_search_clause:
        return _run_vector_query_cypher25(question, top_k, query_embedding=query_embedding)
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


def _run_vector_query_cypher25(
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


def _run_graph_query(question: str, top_k: int) -> list[dict]:  # pragma: no cover
    """Execute graph-based entity search."""
    try:
        q1 = (question or "").strip()
        q2 = _strip_accents(q1)
        queries = [q1]
        if q2 and q2 != q1:
            queries.append(q2)

        rows: list[dict] = []
        with neo4j_client.session() as session:
            for q in queries:
                records = session.run(_GRAPH_CYPHER, q=q, top_k=top_k)
                rows.extend(dict(r) for r in records)

        best: dict[str, dict] = {}
        for r in rows:
            cid = r.get("chunk_id")
            if not cid:
                continue
            if cid not in best or (r.get("graph_score", 0) > best[cid].get("graph_score", 0)):
                best[cid] = r
        rows = list(best.values())
        rows.sort(key=lambda x: x.get("graph_score", 0), reverse=True)
        rows = rows[:top_k]
        logger.info("Graph retrieval executed", extra={"rows": len(rows)})
        return rows
    except Exception as e:
        logger.warning(f"Graph search failed: {e}")
        return []


def _community_search(
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


def _wrrf_fusion(
    results_list: list[list[dict]],
    weights: list[float],
    k: int = 60,
) -> list[dict]:
    """Weighted Reciprocal Rank Fusion.

    Formula: score(d) = sum(weight_i / (k + rank_i(d))) for each signal i
    Deduplicates by chunk_id, keeps highest fused score.
    """
    # Build rank maps for each signal (1-based ranks)
    rank_maps: list[dict[str, int]] = []
    for results in results_list:
        ranks = {r["chunk_id"]: i + 1 for i, r in enumerate(results)}
        rank_maps.append(ranks)

    # Collect all chunk IDs across all signals
    all_chunk_ids: set[str] = set()
    for rm in rank_maps:
        all_chunk_ids.update(rm.keys())

    # Compute WRRF scores
    wrrf_scores: dict[str, float] = {}
    for chunk_id in all_chunk_ids:
        score = 0.0
        for weight, rm in zip(weights, rank_maps):
            if chunk_id in rm:
                score += weight / (k + rm[chunk_id])
        wrrf_scores[chunk_id] = score

    # Build result metadata map (first occurrence wins)
    result_map: dict[str, dict] = {}
    for results in results_list:
        for r in results:
            chunk_id = r["chunk_id"]
            if chunk_id not in result_map:
                result_map[chunk_id] = {
                    "page_title": r.get("page_title", ""),
                    "page_url": r.get("page_url", ""),
                    "page_id": r.get("page_id", ""),
                    "chunk_id": chunk_id,
                    "chunk_text": r.get("chunk_text", ""),
                }

    # Sort by WRRF score descending
    sorted_results = sorted(
        [{**result_map[cid], "score": wrrf_scores[cid]} for cid in wrrf_scores],
        key=lambda x: x["score"],
        reverse=True,
    )

    logger.info(
        "WRRF fusion completed",
        extra={
            "signals": len(results_list),
            "candidates": len(all_chunk_ids),
            "output": len(sorted_results),
        },
    )

    return sorted_results


def _vector_search(query_embedding: list[float], top_k: int) -> list[dict]:
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


def _graph_search(query: str, top_k: int) -> list[dict]:
    """Entity-based graph traversal: find entities mentioned in query, return their chunks.

    Uses the entity_alias_ft fulltext index to match query terms against entity names,
    then traverses MENTIONS edges to retrieve associated chunks.

    Args:
        query: Natural language query string.
        top_k: Maximum number of results to return.

    Returns:
        List of dicts with keys: page_title, page_url, page_id, chunk_id, chunk_text, graph_score.
    """
    try:
        with neo4j_client.session() as session:
            records = session.run(_GRAPH_CYPHER, q=query, top_k=top_k)
            rows = [dict(r) for r in records]
        logger.info("_graph_search executed", extra={"rows": len(rows)})
        return rows
    except Exception as e:
        logger.warning(f"_graph_search failed: {e}")
        return []


def _wrrf_fuse(
    results_by_channel: dict[str, list[dict]],
    k: int = 60,
) -> list[dict]:
    """Weighted Reciprocal Rank Fusion over named retrieval channels.

    Formula: score(d) = sum(weight_i / (k + rank_i(d))) for each channel i
    where rank_i(d) is the 1-based rank of document d in channel i.

    Channel weights are read from settings:
      - bm25    -> settings.wrrf_weight_bm25
      - vector  -> settings.wrrf_weight_vector
      - graph   -> settings.wrrf_weight_graph
      - community -> settings.wrrf_weight_community

    Args:
        results_by_channel: Mapping of channel name to ranked result list.
            Each result dict must contain at least 'chunk_id' and 'chunk_text'.
        k: RRF smoothing constant (default 60). Higher k reduces the influence
            of high ranks; lower k amplifies top-ranked documents.

    Returns:
        Deduplicated list of result dicts sorted by fused score descending.
        Each dict contains: page_title, page_url, page_id, chunk_id, chunk_text, score,
        and a 'channel_scores' dict showing per-channel contribution.
    """
    weight_map = {
        "bm25": settings.wrrf_weight_bm25,
        "vector": settings.wrrf_weight_vector,
        "graph": settings.wrrf_weight_graph,
        "community": settings.wrrf_weight_community,
    }

    # Build per-channel rank maps (1-based)
    channel_ranks: dict[str, dict[str, int]] = {}
    for channel_name, results in results_by_channel.items():
        ranks: dict[str, int] = {}
        for i, r in enumerate(results):
            cid = r.get("chunk_id")
            if cid and cid not in ranks:
                ranks[cid] = i + 1
        channel_ranks[channel_name] = ranks

    # Collect all unique chunk IDs
    all_chunk_ids: set[str] = set()
    for ranks in channel_ranks.values():
        all_chunk_ids.update(ranks.keys())

    if not all_chunk_ids:
        return []

    # Compute fused scores
    fused_scores: dict[str, float] = {}
    channel_contributions: dict[str, dict[str, float]] = {}
    for chunk_id in all_chunk_ids:
        total_score = 0.0
        contributions: dict[str, float] = {}
        for channel_name, ranks in channel_ranks.items():
            if chunk_id in ranks:
                weight = weight_map.get(channel_name, 0.0)
                contribution = weight / (k + ranks[chunk_id])
                total_score += contribution
                contributions[channel_name] = contribution
        fused_scores[chunk_id] = total_score
        channel_contributions[chunk_id] = contributions

    # Build metadata map from first occurrence across channels
    result_meta: dict[str, dict] = {}
    for results in results_by_channel.values():
        for r in results:
            cid = r.get("chunk_id")
            if cid and cid not in result_meta:
                result_meta[cid] = {
                    "page_title": r.get("page_title", ""),
                    "page_url": r.get("page_url", ""),
                    "page_id": r.get("page_id", ""),
                    "chunk_id": cid,
                    "chunk_text": r.get("chunk_text", ""),
                }

    # Assemble and sort
    fused_results = []
    for cid in fused_scores:
        entry = {**result_meta.get(cid, {"chunk_id": cid})}
        entry["score"] = fused_scores[cid]
        entry["channel_scores"] = channel_contributions[cid]
        fused_results.append(entry)

    fused_results.sort(key=lambda x: x["score"], reverse=True)

    logger.info(
        "wRRF fusion completed",
        extra={
            "channels": list(results_by_channel.keys()),
            "candidates": len(all_chunk_ids),
            "k": k,
        },
    )

    return fused_results


def hybrid_retrieve(query: str, top_k: int = 10) -> list[dict]:
    """Hybrid retrieval combining BM25, vector, and graph channels via weighted RRF.

    This is the primary public retrieval function for the API layer. It:
    1. Embeds the query once (shared across vector and community channels).
    2. Runs BM25 fulltext search, vector similarity search, and graph entity search.
    3. Optionally includes community search if Community nodes exist.
    4. Fuses all channels via weighted Reciprocal Rank Fusion.
    5. Returns the top_k results sorted by fused score.

    Args:
        query: Natural language question in Vietnamese.
        top_k: Number of final results to return (default 10).

    Returns:
        List of result dicts with keys: page_title, page_url, page_id, chunk_id,
        chunk_text, score, channel_scores.
    """
    # Embed query once for vector and community channels
    query_embedding: list[float] | None = None
    try:
        query_embedding = embed_texts([query])[0]
    except Exception as e:
        logger.warning(f"hybrid_retrieve: embedding failed, vector/community disabled: {e}")

    # Fetch candidates from each channel (over-fetch for better fusion)
    candidate_k = top_k * 3

    bm25_results = _run_bm25_query(query, candidate_k)
    vector_results = _vector_search(query_embedding, candidate_k) if query_embedding else []
    graph_results = _graph_search(query, candidate_k)

    # Build channel map
    results_by_channel: dict[str, list[dict]] = {}
    if bm25_results:
        results_by_channel["bm25"] = bm25_results
    if vector_results:
        results_by_channel["vector"] = vector_results
    if graph_results:
        results_by_channel["graph"] = graph_results

    # Optionally add community channel
    if query_embedding:
        community_results = _community_search(query, top_k * 2, query_embedding=query_embedding)
        if community_results:
            results_by_channel["community"] = community_results

    # If no channel returned results, return empty
    if not results_by_channel:
        logger.warning("hybrid_retrieve: all channels returned empty results")
        return []

    # Fuse via wRRF
    fused = _wrrf_fuse(results_by_channel, k=settings.wrrf_k)

    return fused[:top_k]


def _run_fallback_query(question: str, top_k: int) -> list[dict]:
    """Execute WRRF hybrid search as fallback when LLM-generated Cypher fails.

    Combines BM25, vector, graph, and community signals via Weighted Reciprocal Rank Fusion.
    Falls back to legacy hybrid Cypher if all signals fail.
    """
    # Embed question once for vector and community signals
    try:
        query_embedding = embed_texts([question])[0]
    except Exception:
        query_embedding = None

    bm25_results = _run_bm25_query(question, top_k * 3)
    title_results = _run_title_query(question, top_k * 3)
    vector_results = (
        _run_vector_query(question, top_k * 3, query_embedding=query_embedding)
        if query_embedding
        else []
    )
    graph_results = _run_graph_query(question, top_k * 3)
    community_results = (
        _community_search(question, top_k * 2, query_embedding=query_embedding)
        if query_embedding
        else []
    )

    all_results = [bm25_results, title_results, vector_results, graph_results]
    # Title matches are useful but can be broad; keep a smaller weight.
    all_weights = [
        settings.wrrf_weight_bm25,
        settings.wrrf_weight_bm25 * 0.6,
        settings.wrrf_weight_vector,
        settings.wrrf_weight_graph,
    ]

    if community_results:
        all_results.append(community_results)
        all_weights.append(settings.wrrf_weight_community)

    if not any(all_results):
        # All signals failed; fall back to legacy single-query hybrid
        logger.warning("WRRF signals all empty, using legacy hybrid fallback")
        return _run_legacy_fallback_query(question, top_k)

    fused = _wrrf_fusion(
        all_results,
        all_weights,
        k=settings.wrrf_k,
    )
    return fused[:top_k]


def _run_legacy_fallback_query(question: str, top_k: int) -> list[dict]:
    """Execute legacy hybrid UNION query as last-resort fallback."""
    with neo4j_client.session() as session:
        records = session.run(
            _LEGACY_HYBRID_CYPHER,
            q=question,
            top_k=top_k,
        )
        rows = [dict(r) for r in records]
    logger.info("Legacy fallback retrieval executed", extra={"rows": len(rows)})
    return rows


def _expand_via_links(page_ids: list[str], question: str, top_k: int) -> list[dict]:
    """Retrieve chunks from pages connected via LINKS_TO edges."""
    if not page_ids:
        return []
    with neo4j_client.session() as session:
        records = session.run(
            _EXPAND_LINKS_CYPHER,
            page_ids=page_ids,
            q=question,
            top_k=top_k,
        )
        rows = [dict(r) for r in records]
    logger.info("Multi-hop expansion", extra={"source_pages": len(page_ids), "expanded_rows": len(rows)})
    return rows


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _fix_mojibake(s: str) -> str:
    """
    Best-effort fix for common UTF-8-as-Latin1 mojibake (e.g. 'Minh MÃ¡ng').
    If it doesn't look like mojibake or decoding fails, return original.
    """
    s = "" if s is None else str(s)
    # Heuristic markers seen in typical mojibake for Vietnamese/UTF-8 text on Windows.
    # We intentionally include a few broad markers ('Ä', 'Æ') that show up in
    # UTF-8 mis-decoded as Latin-1/CP1252 (e.g. 'Ä' for 'đ').
    markers = ("Ã", "Â", "â", "Ä", "Æ", "Å", "áº", "á»", "áť", "º", "»", "¿")
    if not any(m in s for m in markers):
        return s

    def _repair(codec: str) -> str:
        try:
            return s.encode(codec, errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            return ""

    # Try the most common mappings first.
    candidates = [_repair("latin1"), _repair("cp1252")]
    candidates = [c for c in candidates if c]
    if not candidates:
        return s

    def _has_vietnamese(t: str) -> bool:
        # Vietnamese-specific letters live in a few blocks:
        # - Latin-1 Supplement: a lot of accented vowels (à, á, â, ê, ô, ...)
        # - Latin Extended-A/B: ă, đ, ơ, ư (and uppercase variants)
        # - Vietnamese block: ạ, ả, ấ, ề, ễ, ộ, ỵ, ...
        if any(ch in t for ch in ("đ", "Đ", "ă", "Ă", "ơ", "Ơ", "ư", "Ư")):
            return True
        return any(
            (0x00C0 <= ord(ch) <= 0x00FF) or (0x1EA0 <= ord(ch) <= 0x1EFF)
            for ch in t
        )

    # Prefer candidates that "recover" Vietnamese codepoints.
    for c in candidates:
        if _has_vietnamese(c):
            return c

    # Otherwise, pick the candidate that most reduces mojibake markers, but avoid
    # returning a heavily truncated string (which can happen with error="ignore").
    def _score(t: str) -> int:
        return sum(t.count(m) for m in markers)

    best = min(candidates, key=_score)
    if best and (len(best) >= int(len(s) * 0.9)):
        return best
    return s


def _extractive_fallback(snippets: list[dict]) -> str:
    """High-precision fallback: quote sentences directly from retrieved context."""
    if not snippets:
        return "Không tìm thấy thông tin liên quan trong dữ liệu hiện có."

    lines: list[str] = []
    seen_sentences: set[str] = set()
    for s in snippets[:3]:
        idx = s["idx"]
        sentences = _split_sentences(s["text"])
        if sentences:
            sent = sentences[0]
            key = re.sub(r"\s+", " ", sent.strip().lower())
            if key and key not in seen_sentences:
                seen_sentences.add(key)
                lines.append(f"- {sent} [{idx}]")
        else:
            preview = (s["text"][:200] or "").strip()
            key = re.sub(r"\s+", " ", preview.strip().lower())
            if key and key not in seen_sentences:
                seen_sentences.add(key)
                lines.append(f"- {preview}... [{idx}]")

    if not lines:
        return "Không tìm thấy thông tin liên quan trong dữ liệu hiện có."

    return "Dựa trên các đoạn trích từ Wikipedia:\n" + "\n".join(lines)
def _synthesize_answer_grounded(question: str, snippets: list[dict]) -> str:
    """Grounded synthesis: only use provided snippets and always cite."""
    if not snippets:
        return "Không tìm thấy thông tin liên quan trong dữ liệu hiện có."

    context = "\n\n".join(f"[{s['idx']}] {s['text']}" for s in snippets)
    prompt = (
        "Bạn là trợ lý tra cứu Wikipedia.\n"
        "QUY TẮC BẮT BUỘC:\n"
        "- Chỉ được dùng thông tin có trong Ngữ cảnh.\n"
        "- Mọi khẳng định phải có trích dẫn dạng [1], [2]...\n"
        "- Nếu Ngữ cảnh không đủ để trả lời chắc chắn: trả lời 'Không đủ thông tin trong dữ liệu hiện có.'\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        "Trả lời (ngắn gọn, đúng trọng tâm, có trích dẫn):"
    )

    try:
        if settings.model_mode == "ollama":
            from src.infrastructure.ollama_llm import chat as ollama_chat

            answer = ollama_chat(
                [
                    {"role": "system", "content": "Bạn là trợ lý hỏi đáp tiếng Việt."},
                    {"role": "user", "content": prompt},
                ],
                max_new_tokens=350,
                temperature=0.0,
            )
            if answer:
                return answer

        from google.genai import types

        clients = _client_pool()
        for client in clients:
            try:
                resp = client.models.generate_content(
                    model=settings.gemini_model_text,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=300),
                )
                answer = (resp.text or "").strip()
                if answer:
                    return answer
            except Exception:
                continue
    except Exception as exc:
        logger.warning("LLM synthesis failed, using extractive fallback", extra={"error": str(exc)})

    return _extractive_fallback(snippets)


def _synthesize_answer(question: str, snippets: list[str]) -> str:
    """Synthesize a natural-language answer from retrieved snippets using LLM."""
    if not snippets:
        return "Không tìm thấy thông tin liên quan."

    context = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(snippets))
    prompt = (
        f"Dựa vào các đoạn văn bản sau, hãy trả lời câu hỏi một cách ngắn gọn và chính xác bằng tiếng Việt.\n\n"
        f"Câu hỏi: {question}\n\n"
        f"Ngữ cảnh:\n{context}\n\n"
        f"Trả lời:"
    )

    try:
        if settings.model_mode == "ollama":
            from src.infrastructure.ollama_llm import chat as ollama_chat

            answer = ollama_chat(
                [
                    {"role": "system", "content": "Bạn là trợ lý hỏi đáp tiếng Việt."},
                    {"role": "user", "content": prompt},
                ],
                max_new_tokens=350,
                temperature=0.2,
            )
            if answer:
                return answer

        from google.genai import types

        clients = _client_pool()
        for client in clients:
            try:
                resp = client.models.generate_content(
                    model=settings.gemini_model_text,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=300),
                )
                answer = (resp.text or "").strip()
                if answer:
                    return answer
            except Exception:
                continue
    except Exception as exc:
        logger.warning("LLM synthesis failed, using snippet fallback", extra={"error": str(exc)})

    return "Dựa trên thông tin tìm được: " + " | ".join(s[:200] for s in snippets[:2])


def _run_generated_query(question: str, top_k: int) -> list[dict]:
    """Generate, validate, and execute LLM-produced read-only Cypher."""
    cypher = generate_readonly_cypher(question)
    assert_readonly_cypher(cypher)

    with neo4j_client.session() as session:
        records = session.run(cypher, q=question, top_k=top_k)
        rows = [dict(r) for r in records]

    required_keys = {"page_title", "page_url", "chunk_id", "chunk_text", "score"}
    for row in rows:
        if not required_keys.issubset(row):
            raise RuntimeError("Generated query returned unexpected shape")

    logger.info("Generated retrieval executed", extra={"rows": len(rows)})
    return rows


def _run_template_query(question: str, top_k: int) -> tuple[str | None, list[dict]]:
    """Run a stable, intent-specific Cypher template when possible.

    Returns (template_intent, rows).
    """
    p = plan_query(question)
    cypher = pick_template(p.intent)
    if not cypher:
        return None, []

    q = build_fulltext_query(question, subject=p.subject)
    with neo4j_client.session() as session:
        records = session.run(cypher, q=q, top_k=min(8, max(1, top_k)))
        rows = [dict(r) for r in records]

    # Must match downstream expectations.
    required_keys = {"page_title", "page_url", "page_id", "chunk_id", "chunk_text", "score"}
    rows = [r for r in rows if required_keys.issubset(r)]
    return p.intent, rows


def query_graph(question: str, top_k: int = 4) -> QueryResult:
    """Query graph and synthesize a deterministic answer with citations."""
    if settings.model_mode == "local":
        from src.orchestration.agent import agent_query

        return agent_query(question, top_k)

    # Fast path: if the question looks like it's about a specific Page title,
    # resolve it accent-insensitively and pull chunks directly from that page.
    page = None
    try:
        page = _resolve_page_by_title(question)
    except Exception:
        page = None

    if page and page.get("id"):
        retrieval_tier = "page_title"
        with neo4j_client.session() as session:
            records = session.run(_PAGE_CHUNKS_CYPHER, page_id=page["id"], top_k=top_k * 5)
            rows = [dict(r) for r in records]
    else:
        # Prefer deterministic templates over free-form LLM Cypher generation.
        template_intent, rows = _run_template_query(question, top_k)
        if rows:
            retrieval_tier = f"template:{template_intent}"
        else:
            retrieval_tier = "generated"
            try:
                rows = _run_generated_query(question, top_k)
            except (RuntimeError, ValueError, KeyError, TypeError, CypherSyntaxError) as exc:
                logger.warning("Generated query failed, falling back to WRRF", extra={"error": str(exc)})
                retrieval_tier = "wrrf"
                rows = _run_fallback_query(question, top_k)

    if not rows:
        return QueryResult(
            answer="Không đủ thông tin trong dữ liệu hiện có.",
            citations=[],
            retrieval_tier=retrieval_tier,
        )

    from src.retrieval.reranker import rerank

    reranked = rerank(question, rows, text_key="chunk_text", top_k=top_k)

    # Safety filter: if retrieval is off-topic, prefer abstaining over hallucinating.
    q_terms = _key_terms(question)
    if q_terms:
        filtered = []
        for r in reranked:
            title_terms = set(_normalize_for_tokens(r.get("page_title", "")).split())
            if q_terms & title_terms:
                filtered.append(r)
        if filtered:
            reranked = filtered
        else:
            return QueryResult(
                answer="Không đủ thông tin trong dữ liệu hiện có.",
                citations=[],
                retrieval_tier=retrieval_tier,
            )

    if settings.multi_hop_expansion and reranked:
        page_ids = list({r["page_id"] for r in reranked if r.get("page_id")})
        expanded = _expand_via_links(page_ids, question, top_k)
        if expanded:
            seen_chunks = {r["chunk_id"] for r in reranked}
            new_rows = [r for r in expanded if r["chunk_id"] not in seen_chunks]
            if new_rows:
                combined = reranked + new_rows
                reranked = rerank(question, combined, text_key="chunk_text", top_k=top_k)

    # Avoid repeated answers/citations when multiple chunks come from the same page.
    limited: list[dict] = []
    per_page: dict[str, int] = {}
    for r in reranked:
        url = str(r.get("page_url") or "")
        per_page[url] = per_page.get(url, 0) + 1
        if per_page[url] <= 2:
            limited.append(r)
        if len(limited) >= max(top_k, 4):
            break

    citations = []
    seen_pages: set[str] = set()
    for r in limited:
        url = _fix_mojibake(str(r.get("page_url") or ""))
        if not url or url in seen_pages:
            continue
        seen_pages.add(url)
        citations.append({
            "page_title": _fix_mojibake(r.get("page_title", "")),
            "page_url": url,
            "chunk_id": r.get("chunk_id", ""),
        })

    # Heuristic: for questions explicitly asking "hiệp ước/hiệp định nào", the
    # answer is often exactly the retrieved treaty page title. If we have such a
    # citation, return it directly to avoid the LLM abstaining when snippets are
    # missing the exact phrasing.
    q_norm = (question or "").lower()
    if citations and ("hiệp ước nào" in q_norm or "hiệp định nào" in q_norm):
        for i, c in enumerate(citations, start=1):
            title = (c.get("page_title") or "").strip()
            if title.startswith("Hiệp định") or title.startswith("Hiệp ước"):
                return QueryResult(
                    answer=f"{title} [{i}].",
                    citations=citations,
                    retrieval_tier=retrieval_tier,
                )

    snippets = []
    for i, r in enumerate(limited[:3], start=1):
        txt = _fix_mojibake((r["chunk_text"] or "").strip()).replace("\n", " ")
        snippets.append({"idx": i, "text": txt[:700]})

    synthesized = _fix_mojibake(_synthesize_answer_grounded(question, snippets))
    # If the model abstains but we do have retrieved evidence, fall back to a
    # high-precision extractive answer instead of returning "no info".
    if synthesized.strip().startswith("Không đủ thông tin"):
        synthesized = _extractive_fallback(snippets)
    answer = synthesized

    return QueryResult(answer=answer, citations=citations, retrieval_tier=retrieval_tier)
