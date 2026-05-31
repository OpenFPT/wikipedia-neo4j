"""Build community structure from entity co-occurrence graph using Leiden detection.

Exports entity co-occurrence from Neo4j, runs Leiden clustering via igraph/leidenalg,
writes community_id back to Entity nodes, generates summaries, and saves to JSONL.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import igraph as ig
import leidenalg

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


def build_entity_graph(limit: int | None = None) -> tuple[ig.Graph, list[str]]:
    """Build co-occurrence graph: entities connected if they appear in the same chunk.

    Returns an igraph Graph and a list mapping vertex indices to entity IDs.
    """
    logger.info("Building entity co-occurrence graph from Neo4j...")

    edges: list[tuple[str, str]] = []
    weights: list[int] = []

    query = """
        MATCH (e1:Entity)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:Entity)
        WHERE elementId(e1) < elementId(e2)
        RETURN e1.id AS source, e2.id AS target, count(c) AS weight
    """
    if limit:
        query += f"\nLIMIT {limit * 50}"

    with neo4j_client.session() as session:
        records = session.run(query)
        for r in records:
            edges.append((r["source"], r["target"]))
            weights.append(r["weight"])

    if not edges:
        logger.warning("No co-occurrence edges found")
        return ig.Graph(), []

    # Build vertex set preserving order
    vertex_set: dict[str, int] = {}
    for src, tgt in edges:
        if src not in vertex_set:
            vertex_set[src] = len(vertex_set)
        if tgt not in vertex_set:
            vertex_set[tgt] = len(vertex_set)

    vertex_ids = [""] * len(vertex_set)
    for entity_id, idx in vertex_set.items():
        vertex_ids[idx] = entity_id

    # Build igraph
    edge_indices = [(vertex_set[s], vertex_set[t]) for s, t in edges]
    G = ig.Graph(n=len(vertex_set), edges=edge_indices, directed=False)
    G.es["weight"] = weights
    G.vs["entity_id"] = vertex_ids

    logger.info(
        f"Entity graph built: {G.vcount()} nodes, {G.ecount()} edges"
    )
    return G, vertex_ids


def detect_communities(
    G: ig.Graph, resolution: float = 1.0, min_size: int = 3
) -> list[list[int]]:
    """Run Leiden community detection and filter by minimum size.

    Returns list of communities, each a list of vertex indices.
    """
    if G.vcount() == 0:
        logger.warning("Empty graph, no communities to detect")
        return []

    logger.info(f"Running Leiden community detection (resolution={resolution})...")

    partition = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
        n_iterations=-1,
    )

    # Group vertex indices by community
    community_map: dict[int, list[int]] = {}
    for vertex_idx, comm_id in enumerate(partition.membership):
        community_map.setdefault(comm_id, []).append(vertex_idx)

    # Filter by min size and sort by size descending
    filtered = [members for members in community_map.values() if len(members) >= min_size]
    filtered.sort(key=len, reverse=True)

    logger.info(
        f"Detected {len(community_map)} communities, "
        f"{len(filtered)} with >= {min_size} members"
    )
    return filtered


def _get_community_context(
    entity_ids: list[str], max_entities: int = 20, max_passages: int = 5
) -> tuple[list[str], list[str]]:
    """Fetch entity names and sample passages for a community."""
    id_list = entity_ids[: max_entities * 2]

    with neo4j_client.session() as session:
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


def write_community_ids_to_entities(
    communities: list[list[int]], vertex_ids: list[str], dry_run: bool = False
) -> int:
    """Write community_id property back to Entity nodes in Neo4j.

    Returns total entities updated.
    """
    total = 0
    batch_size = 500

    for comm_idx, members in enumerate(communities):
        entity_ids = [vertex_ids[v] for v in members]

        if dry_run:
            total += len(entity_ids)
            continue

        for i in range(0, len(entity_ids), batch_size):
            batch = entity_ids[i : i + batch_size]
            with neo4j_client.session() as session:
                session.run(
                    """
                    UNWIND $ids AS eid
                    MATCH (e:Entity {id: eid})
                    SET e.community_id = $community_id
                    """,
                    ids=batch,
                    community_id=comm_idx,
                )
            total += len(batch)

    logger.info(f"Updated community_id on {total} entity nodes (dry_run={dry_run})")
    return total


def store_communities(
    communities: list[dict],
    batch_size: int = 100,
    dry_run: bool = False,
) -> int:
    """Write Community nodes and HAS_MEMBER relationships to Neo4j."""
    if dry_run:
        logger.info(f"[DRY RUN] Would store {len(communities)} Community nodes")
        return len(communities)

    total = 0

    for i in range(0, len(communities), batch_size):
        batch = communities[i : i + batch_size]

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


def save_summaries_jsonl(communities: list[dict], output_path: Path) -> None:
    """Save community summaries to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in communities:
            record = {
                "community_id": c["id"],
                "level": c["level"],
                "summary": c["summary"],
                "member_count": c["member_count"],
                "top_entities": c.get("top_entities", []),
                "entity_ids": c["entity_ids"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(communities)} community summaries to {output_path}")


def run(
    limit: int | None = None,
    min_size: int = 3,
    batch_size: int = 10,
    resolution: float = 1.0,
    dry_run: bool = False,
) -> None:
    """Main pipeline: detect communities, summarize, embed, store."""
    checkpoint_path = Path("data/export/communities.checkpoint.json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    summaries_path = Path("data/communities/summaries.jsonl")
    summaries_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = _load_checkpoint(checkpoint_path)
    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} communities already processed")

    # Step 1: Build graph and detect communities with Leiden
    G, vertex_ids = build_entity_graph(limit=limit)
    communities_indices = detect_communities(G, resolution=resolution, min_size=min_size)

    if limit:
        communities_indices = communities_indices[:limit]

    logger.info(f"Processing {len(communities_indices)} communities...")

    # Step 2: Write community_id back to Entity nodes
    write_community_ids_to_entities(communities_indices, vertex_ids, dry_run=dry_run)

    # Step 3: Generate summaries and embeddings
    results: list[dict] = []
    processed_count = 0

    for idx, member_indices in enumerate(communities_indices):
        if _STOP:
            logger.info("Stopping due to SIGINT")
            break

        community_id = f"community_{idx:05d}"

        if community_id in processed_ids:
            continue

        entity_ids = [vertex_ids[v] for v in member_indices]

        # Get context for this community
        entities, passages = _get_community_context(entity_ids)

        if not entities:
            logger.debug(f"Skipping {community_id}: no entity names found")
            processed_ids.add(community_id)
            continue

        # Generate summary
        if dry_run:
            summary = ", ".join(entities[:10])
        else:
            try:
                summary = generate_summary(entities, passages)
            except Exception as exc:
                logger.warning(f"Summary generation failed for {community_id}: {exc}")
                summary = ", ".join(entities[:10])

        # Generate embedding
        embedding: list[float] | None = None
        if not dry_run:
            try:
                embedding = embed_texts([summary])[0]
            except Exception as exc:
                logger.warning(f"Embedding failed for {community_id}: {exc}")

        results.append(
            {
                "id": community_id,
                "level": 0,
                "summary": summary,
                "embedding": embedding,
                "member_count": len(entity_ids),
                "entity_ids": entity_ids,
                "top_entities": entities[:10],
            }
        )

        processed_ids.add(community_id)
        processed_count += 1

        # Batch store and checkpoint
        if len(results) >= batch_size:
            stored = store_communities(results, dry_run=dry_run)
            logger.info(f"Stored {stored} communities (total processed: {processed_count})")
            if not dry_run:
                _save_checkpoint(checkpoint_path, processed_ids)
            results = []

        # Rate limit for API calls
        if not dry_run and settings.model_mode != "local" and settings.embedding_backend != "local":
            time.sleep(0.5)

    # Store remaining
    if results:
        stored = store_communities(results, dry_run=dry_run)
        logger.info(f"Stored {stored} remaining communities")
        if not dry_run:
            _save_checkpoint(checkpoint_path, processed_ids)

    # Save summaries to JSONL for the community retrieval module
    if results or processed_count > 0:
        save_summaries_jsonl(
            [r for r in results if r.get("summary")],
            summaries_path,
        )

    logger.info(
        f"Community detection complete. Processed {processed_count} new communities. "
        f"(dry_run={dry_run})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build community structure from entity co-occurrence graph using Leiden"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max communities to process")
    parser.add_argument("--min-size", type=int, default=3, help="Min entities per community")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for Neo4j writes")
    parser.add_argument(
        "--resolution", type=float, default=1.0, help="Leiden resolution parameter"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing to Neo4j or calling LLM"
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    logger.info(
        "Starting Leiden community detection",
        extra={
            "limit": args.limit,
            "min_size": args.min_size,
            "batch_size": args.batch_size,
            "resolution": args.resolution,
            "dry_run": args.dry_run,
        },
    )

    run(
        limit=args.limit,
        min_size=args.min_size,
        batch_size=args.batch_size,
        resolution=args.resolution,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
