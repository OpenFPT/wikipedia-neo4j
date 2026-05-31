"""Batch relation extraction from all chunks in Neo4j."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.logging_utils import configure_logging, get_logger
from src.infrastructure.neo4j_client import neo4j_client
from src.extraction.relations import RELATION_TYPES, extract_relations

configure_logging(settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="relations")
logger = get_logger(__name__)

_STOP = False


def _handle_sigint(sig, frame):
    global _STOP
    _STOP = True
    print("\nGraceful shutdown requested, finishing current batch...")


def _load_checkpoint(checkpoint_path: Path) -> str | None:
    """Load last processed chunk_id from checkpoint."""
    if checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text())
        return data.get("last_chunk_id")
    return None


def _save_checkpoint(checkpoint_path: Path, last_chunk_id: str, stats: dict) -> None:
    """Atomically save checkpoint."""
    tmp = checkpoint_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"last_chunk_id": last_chunk_id, **stats}, indent=2))
    tmp.rename(checkpoint_path)


def _fetch_chunks(last_id: str | None, limit: int | None, batch_size: int):
    """Fetch chunks from Neo4j ordered by id for resumability."""
    if last_id:
        cypher = (
            "MATCH (c:Chunk) WHERE c.id > $last_id "
            "RETURN c.id AS chunk_id, c.text AS text "
            "ORDER BY c.id LIMIT $batch_size"
        )
    else:
        cypher = (
            "MATCH (c:Chunk) "
            "RETURN c.id AS chunk_id, c.text AS text "
            "ORDER BY c.id LIMIT $batch_size"
        )

    total_fetched = 0
    current_last_id = last_id

    while True:
        if _STOP:
            break
        if limit is not None and total_fetched >= limit:
            break

        effective_batch = batch_size
        if limit is not None:
            effective_batch = min(batch_size, limit - total_fetched)

        with neo4j_client.session() as session:
            if current_last_id:
                result = session.run(cypher, last_id=current_last_id, batch_size=effective_batch)
            else:
                result = session.run(cypher, batch_size=effective_batch)
            records = [{"chunk_id": r["chunk_id"], "text": r["text"]} for r in result]

        if not records:
            break

        yield records
        total_fetched += len(records)
        current_last_id = records[-1]["chunk_id"]

        # Update cypher to use the cursor pattern
        cypher = (
            "MATCH (c:Chunk) WHERE c.id > $last_id "
            "RETURN c.id AS chunk_id, c.text AS text "
            "ORDER BY c.id LIMIT $batch_size"
        )


def _load_relations_to_neo4j(triples_by_type: dict[str, list[dict]]) -> int:
    """Load extracted relations into Neo4j as typed edges."""
    total = 0
    for rel_type in RELATION_TYPES:
        rows = triples_by_type.get(rel_type, [])
        if not rows:
            continue
        # Use per-type MERGE query (no APOC dependency)
        cypher = (
            "UNWIND $rows AS row "
            "MATCH (e1:Entity) WHERE e1.name = row.subject "
            "MATCH (e2:Entity) WHERE e2.name = row.object "
            f"MERGE (e1)-[:{rel_type} {{source: 'llm_extract'}}]->(e2)"
        )
        count = neo4j_client.run_batch(cypher, rows, batch_size=500)
        total += count
        logger.info("Loaded relations", extra={"type": rel_type, "count": count})
    return total


def main():
    parser = argparse.ArgumentParser(description="Batch relation extraction from Neo4j chunks")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to process")
    parser.add_argument("--batch-size", type=int, default=50, help="Chunks per batch")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini instead of local model")
    parser.add_argument("--output-dir", type=str, default="data/export", help="Output directory")
    parser.add_argument("--load-neo4j", action="store_true", help="Also load relations into Neo4j")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save checkpoint every N chunks")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "relations.jsonl"
    checkpoint_path = output_dir / ".relations_checkpoint.json"

    use_local = not args.use_gemini
    last_chunk_id = _load_checkpoint(checkpoint_path)

    if last_chunk_id:
        print(f"Resuming from checkpoint: chunk_id > {last_chunk_id}")
        mode = "a"
    else:
        mode = "w"

    stats = {"chunks_processed": 0, "triples_extracted": 0, "chunks_failed": 0}
    neo4j_buffer: dict[str, list[dict]] = {rt: [] for rt in RELATION_TYPES}

    start_time = time.time()

    with open(output_file, mode, encoding="utf-8") as fp:
        for batch in _fetch_chunks(last_chunk_id, args.limit, args.batch_size):
            if _STOP:
                break

            for record in batch:
                if _STOP:
                    break

                chunk_id = record["chunk_id"]
                text = record["text"]

                if not text or not text.strip():
                    stats["chunks_processed"] += 1
                    continue

                try:
                    triples = extract_relations(text, use_local=use_local)
                except Exception as e:
                    logger.warning("Extraction failed", extra={"chunk_id": chunk_id, "error": str(e)})
                    stats["chunks_failed"] += 1
                    stats["chunks_processed"] += 1
                    continue

                # Write to JSONL
                row = {
                    "chunk_id": chunk_id,
                    "triples": [
                        {"subject": t.subject, "relation": t.relation, "object": t.object}
                        for t in triples
                    ],
                }
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

                # Buffer for Neo4j loading
                if args.load_neo4j:
                    for t in triples:
                        neo4j_buffer[t.relation].append(
                            {"subject": t.subject, "object": t.object}
                        )

                stats["chunks_processed"] += 1
                stats["triples_extracted"] += len(triples)
                last_chunk_id = chunk_id

                # Progress logging
                if stats["chunks_processed"] % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = stats["chunks_processed"] / elapsed if elapsed > 0 else 0
                    print(
                        f"  Processed: {stats['chunks_processed']:,} chunks | "
                        f"Triples: {stats['triples_extracted']:,} | "
                        f"Rate: {rate:.1f} chunks/s"
                    )

            # Checkpoint after each batch
            if last_chunk_id and stats["chunks_processed"] % args.checkpoint_every < args.batch_size:
                _save_checkpoint(checkpoint_path, last_chunk_id, stats)
                fp.flush()

    # Final checkpoint
    if last_chunk_id:
        _save_checkpoint(checkpoint_path, last_chunk_id, stats)

    # Load into Neo4j if requested
    if args.load_neo4j:
        total_buf = sum(len(v) for v in neo4j_buffer.values())
        if total_buf > 0:
            print(f"\nLoading {total_buf:,} relations into Neo4j...")
            loaded = _load_relations_to_neo4j(neo4j_buffer)
            print(f"  Loaded {loaded:,} relation edges")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Chunks processed: {stats['chunks_processed']:,}")
    print(f"  Triples extracted: {stats['triples_extracted']:,}")
    print(f"  Failures: {stats['chunks_failed']:,}")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
