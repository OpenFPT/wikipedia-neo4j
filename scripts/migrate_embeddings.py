"""Re-embed all existing Chunk nodes with the new embedding model."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.infrastructure.llm import embed_texts_batch
from src.logging_utils import configure_logging, get_logger
from src.infrastructure.neo4j_client import neo4j_client

configure_logging(
    settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="migrate_embeddings"
)
logger = get_logger(__name__)

_STOP = False
CHECKPOINT_FILE = Path("data/migrate_embeddings.checkpoint.json")


def _handle_sigint(sig, frame):
    global _STOP
    _STOP = True
    print("\nGraceful shutdown requested, finishing current batch...")


def _load_checkpoint() -> str | None:
    """Return last processed chunk_id or None."""
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        return data.get("last_chunk_id")
    return None


def _save_checkpoint(last_chunk_id: str) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"last_chunk_id": last_chunk_id}))
    tmp.rename(CHECKPOINT_FILE)


def _fetch_chunks(after_id: str | None, limit: int | None, batch_size: int) -> list[dict]:
    """Fetch chunk nodes ordered by id, optionally after a given id."""
    if after_id:
        query = (
            "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL AND c.id > $after_id "
            "RETURN c.id AS chunk_id, c.text AS text ORDER BY c.id LIMIT $batch"
        )
        params = {"after_id": after_id, "batch": batch_size}
    else:
        query = (
            "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL "
            "RETURN c.id AS chunk_id, c.text AS text ORDER BY c.id LIMIT $batch"
        )
        params = {"batch": batch_size}

    with neo4j_client.session() as session:
        result = session.run(query, **params)
        return [dict(record) for record in result]


def _update_embeddings(rows: list[dict]) -> None:
    """Write new embeddings back to Neo4j."""
    cypher = (
        "UNWIND $rows AS row "
        "MATCH (c:Chunk {id: row.chunk_id}) "
        "SET c.embedding = row.embedding"
    )
    with neo4j_client.session() as session:
        session.run(cypher, rows=rows)


def migrate(limit: int | None, batch_size: int, embed_batch_size: int) -> None:
    last_id = _load_checkpoint()
    if last_id:
        print(f"Resuming after chunk_id: {last_id}")

    total_processed = 0

    while True:
        if _STOP:
            print(f"Stopped. Total re-embedded: {total_processed:,}")
            break

        if limit and total_processed >= limit:
            break

        fetch_size = batch_size
        if limit:
            fetch_size = min(batch_size, limit - total_processed)

        chunks = _fetch_chunks(last_id, limit, fetch_size)
        if not chunks:
            break

        texts = [c["text"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]

        embeddings = embed_texts_batch(texts, batch_size=embed_batch_size)

        update_rows = [
            {"chunk_id": cid, "embedding": emb}
            for cid, emb in zip(chunk_ids, embeddings)
        ]
        _update_embeddings(update_rows)

        last_id = chunk_ids[-1]
        total_processed += len(chunks)
        _save_checkpoint(last_id)

        print(f"  Re-embedded {total_processed:,} chunks (last: {last_id})")

    print(f"\nMigration complete: {total_processed:,} chunks re-embedded.")


def main() -> None:
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(description="Re-embed existing chunks with new model")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to re-embed")
    parser.add_argument("--batch-size", type=int, default=200, help="Chunks per Neo4j fetch")
    parser.add_argument(
        "--embed-batch-size", type=int, default=settings.embed_batch_size, help="Chunks per embedding call"
    )
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and start from beginning")
    args = parser.parse_args()

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint reset.")

    print(f"Model: {settings.local_embedding_model}")
    print(f"Embedding dim: {settings.embedding_dim}")
    print(f"Backend: {settings.embedding_backend}")

    migrate(limit=args.limit, batch_size=args.batch_size, embed_batch_size=args.embed_batch_size)


if __name__ == "__main__":
    main()
