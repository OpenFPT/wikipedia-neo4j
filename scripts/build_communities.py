"""Build community structure from entity co-occurrence graph using Louvain detection.

Generates LLM summaries per community, embeds them, and stores Community nodes in Neo4j.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import louvain_communities

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.llm import embed_texts
from src.logging_utils import configure_logging, get_logger
from src.neo4j_client import neo4j_client

configure_logging(
    settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="communities"
)
logger = get_logger(__name__)

_STOP = False

COMMUNITY_SUMMARY_PROMPT = """Tóm tắt nhóm thực thể liên quan sau đây từ Wikipedia tiếng Việt.

Các thực thể trong nhóm:
{entities}

Đoạn văn bản mẫu:
{passages}

Viết một bản tóm tắt ngắn gọn (2-3 câu) bằng tiếng Việt mô tả nhóm này nói về chủ đề gì và các thực thể liên quan với nhau như thế nào."""


def _handle_sigint(sig: int, frame: object) -> None:
    global _STOP
    _STOP = True
    print("\nGraceful shutdown requested, finishing current batch...")


def _load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load set of already-processed community IDs."""
    if checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text())
        return set(data.get("processed_ids", []))
    return set()


def _save_checkpoint(checkpoint_path: Path, processed_ids: set[str]) -> None:
    """Atomically save checkpoint."""
    tmp = checkpoint_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"processed_ids": sorted(processed_ids)}))
    tmp.rename(checkpoint_path)


def build_entity_graph() -> nx.Graph:
    """Build co-occurrence graph: entities connected if they appear in the same chunk."""
    logger.info("Building entity co-occurrence graph from Neo4j...")
    G = nx.Graph()

    with neo4j_client.session() as session:
        records = session.run(
            """
            MATCH (e1:Entity)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:Entity)
            WHERE elementId(e1) < elementId(e2)
            RETURN e1.id AS source, e2.id AS target, count(c) AS weight
            """
        )
        for r in records:
            G.add_edge(r["source"], r["target"], weight=r["weight"])

    logger.info(f"Entity graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def detect_communities(
    G: nx.Graph, resolution: float = 1.0, min_size: int = 3
) -> list[set[str]]:
    """Run Louvain community detection and filter by minimum size."""
    if G.number_of_nodes() == 0:
        logger.warning("Empty graph, no communities to detect")
        return []

    logger.info(f"Running Louvain community detection (resolution={resolution})...")
    communities = louvain_communities(G, weight="weight", resolution=resolution, seed=42)

    filtered = [c for c in communities if len(c) >= min_size]
    filtered.sort(key=len, reverse=True)

    logger.info(
        f"Detected {len(communities)} communities, {len(filtered)} with >= {min_size} members"
    )
    return filtered


def _get_community_context(entity_ids: set[str], max_entities: int = 20, max_passages: int = 5) -> tuple[list[str], list[str]]:
    """Fetch entity names and sample passages for a community."""
    id_list = sorted(entity_ids)[:max_entities * 2]

    with neo4j_client.session() as session:
        # Get entity names
        entity_records = session.run(
            """
            MATCH (e:Entity)
            WHERE e.id IN $ids
            RETURN e.name AS name, e.type AS type
            ORDER BY e.name
            LIMIT $limit
            """,
            ids=id_list,
            limit=max_entities,
        )
        entities = [f"{r['name']} ({r['type'] or 'Entity'})" for r in entity_records]

        # Get sample passages from chunks mentioning these entities
        passage_records = session.run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE e.id IN $ids
            RETURN DISTINCT c.text AS text
            LIMIT $limit
            """,
            ids=id_list,
            limit=max_passages,
        )
        passages = [r["text"][:300] for r in passage_records if r["text"]]

    return entities, passages


def _generate_summary_gemini(entities: list[str], passages: list[str]) -> str:
    """Generate community summary using Gemini API."""
    from google import genai
    from google.genai import types

    from src.config import load_gemini_api_keys

    prompt = COMMUNITY_SUMMARY_PROMPT.format(
        entities="\n".join(f"- {e}" for e in entities),
        passages="\n---\n".join(passages) if passages else "(Không có đoạn văn mẫu)",
    )

    keys = load_gemini_api_keys()
    last_error: Exception | None = None

    for key in keys:
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=settings.gemini_model_text,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=256,
                ),
            )
            text = (resp.text or "").strip()
            if text:
                return text
            raise RuntimeError("Empty summary response")
        except Exception as exc:
            last_error = exc
            time.sleep(1)
            continue

    raise RuntimeError(f"All Gemini keys failed for summary generation: {last_error}")


def _generate_summary_local(entities: list[str], passages: list[str]) -> str:
    """Generate community summary using local model."""
    from src.local_llm import chat

    prompt = COMMUNITY_SUMMARY_PROMPT.format(
        entities="\n".join(f"- {e}" for e in entities),
        passages="\n---\n".join(passages) if passages else "(Không có đoạn văn mẫu)",
    )

    messages = [
        {"role": "system", "content": "Bạn là trợ lý tóm tắt nội dung Wikipedia tiếng Việt."},
        {"role": "user", "content": prompt},
    ]
    return chat(messages, max_new_tokens=256, temperature=0.3)


def generate_summary(entities: list[str], passages: list[str]) -> str:
    """Generate community summary using configured model backend."""
    if settings.model_mode == "local":
        return _generate_summary_local(entities, passages)
    return _generate_summary_gemini(entities, passages)


def store_communities(
    communities: list[dict],
    batch_size: int = 100,
) -> int:
    """Write Community nodes and HAS_MEMBER relationships to Neo4j."""
    total = 0

    for i in range(0, len(communities), batch_size):
        batch = communities[i : i + batch_size]

        # Create/update Community nodes
        with neo4j_client.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (cm:Community {id: row.id})
                SET cm.level = row.level,
                    cm.summary = row.summary,
                    cm.embedding = row.embedding,
                    cm.member_count = row.member_count
                """,
                rows=[
                    {
                        "id": c["id"],
                        "level": c["level"],
                        "summary": c["summary"],
                        "embedding": c["embedding"],
                        "member_count": c["member_count"],
                    }
                    for c in batch
                ],
            )

        # Create HAS_MEMBER relationships
        for c in batch:
            with neo4j_client.session() as session:
                session.run(
                    """
                    MATCH (cm:Community {id: $community_id})
                    UNWIND $entity_ids AS eid
                    MATCH (e:Entity {id: eid})
                    MERGE (cm)-[:HAS_MEMBER]->(e)
                    """,
                    community_id=c["id"],
                    entity_ids=c["entity_ids"],
                )

        total += len(batch)

    return total


def run(
    limit: int | None = None,
    min_size: int = 3,
    batch_size: int = 10,
    resolution: float = 1.0,
) -> None:
    """Main pipeline: detect communities, summarize, embed, store."""
    checkpoint_path = Path("data/export/communities.checkpoint.json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = _load_checkpoint(checkpoint_path)

    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} communities already processed")

    # Step 1: Build graph and detect communities
    G = build_entity_graph()
    communities_sets = detect_communities(G, resolution=resolution, min_size=min_size)

    if limit:
        communities_sets = communities_sets[:limit]

    logger.info(f"Processing {len(communities_sets)} communities...")

    # Step 2: Generate summaries and embeddings
    results: list[dict] = []
    processed_count = 0

    for idx, entity_ids in enumerate(communities_sets):
        if _STOP:
            logger.info("Stopping due to SIGINT")
            break

        community_id = f"community_{idx:05d}"

        if community_id in processed_ids:
            continue

        # Get context for this community
        entities, passages = _get_community_context(entity_ids)

        if not entities:
            logger.debug(f"Skipping {community_id}: no entity names found")
            processed_ids.add(community_id)
            continue

        # Generate summary
        try:
            summary = generate_summary(entities, passages)
        except Exception as exc:
            logger.warning(f"Summary generation failed for {community_id}: {exc}")
            summary = ", ".join(entities[:10])

        # Generate embedding
        try:
            embedding = embed_texts([summary])[0]
        except Exception as exc:
            logger.warning(f"Embedding failed for {community_id}: {exc}")
            embedding = None

        results.append(
            {
                "id": community_id,
                "level": 0,
                "summary": summary,
                "embedding": embedding,
                "member_count": len(entity_ids),
                "entity_ids": sorted(entity_ids),
            }
        )

        processed_ids.add(community_id)
        processed_count += 1

        # Batch store and checkpoint
        if len(results) >= batch_size:
            stored = store_communities(results)
            logger.info(f"Stored {stored} communities (total processed: {processed_count})")
            _save_checkpoint(checkpoint_path, processed_ids)
            results = []

        # Rate limit for API calls
        if settings.model_mode != "local" and settings.embedding_backend != "local":
            time.sleep(0.5)

    # Store remaining
    if results:
        stored = store_communities(results)
        logger.info(f"Stored {stored} remaining communities")
        _save_checkpoint(checkpoint_path, processed_ids)

    logger.info(f"Community detection complete. Processed {processed_count} new communities.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build community structure from entity co-occurrence graph"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max communities to process")
    parser.add_argument("--min-size", type=int, default=3, help="Min entities per community")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for Neo4j writes")
    parser.add_argument(
        "--resolution", type=float, default=1.0, help="Louvain resolution parameter"
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    logger.info(
        "Starting community detection",
        extra={
            "limit": args.limit,
            "min_size": args.min_size,
            "batch_size": args.batch_size,
            "resolution": args.resolution,
        },
    )

    run(
        limit=args.limit,
        min_size=args.min_size,
        batch_size=args.batch_size,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
