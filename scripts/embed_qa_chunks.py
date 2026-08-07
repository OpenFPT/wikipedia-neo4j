"""Backfill embeddings for Chunk nodes ingested via scripts/ingest_qa_datasets.py.

Queries Neo4j for chunks missing `.embedding`, embeds them via the Gemini API
(embed_texts_batch, key-rotation + retry already handled in
src/infrastructure/llm.py), and writes vectors back through batched UNWIND.
Resumable via a checkpoint file; safe to Ctrl+C and re-run.

Usage:
    uv run python -m scripts.embed_qa_chunks --limit 50
    uv run python -m scripts.embed_qa_chunks
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

from src.config import settings
from src.infrastructure.llm import embed_texts_batch
from src.infrastructure.neo4j_client import neo4j_client
from src.logging_utils import configure_logging, get_logger

configure_logging(settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="embed_qa")
logger = get_logger(__name__)

_CHECKPOINT_PATH = Path(".embed_qa_checkpoint.json")
_STOP = False


def _handle_sigint(_sig, _frame) -> None:
    global _STOP
    _STOP = True
    print("\nGraceful shutdown requested, finishing current batch...")


def _load_checkpoint() -> int:
    if _CHECKPOINT_PATH.exists():
        return json.loads(_CHECKPOINT_PATH.read_text()).get("processed", 0)
    return 0


def _save_checkpoint(processed: int) -> None:
    tmp = _CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps({"processed": processed}))
    tmp.replace(_CHECKPOINT_PATH)


CYPHER_FETCH_MISSING = """
MATCH (c:Chunk)
WHERE c.embedding IS NULL
RETURN c.id AS id, c.text AS text
ORDER BY c.id
LIMIT $limit
"""

CYPHER_WRITE_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (c:Chunk {id: row.chunk_id})
SET c.embedding = row.embedding
"""


def embed_qa_chunks(
    limit: int | None,
    batch_size: int,
    pause_between_batches: float,
    resume: bool,
) -> None:
    settings.embedding_backend = "gemini"

    skip = _load_checkpoint() if resume else 0
    if skip:
        print(f"Resuming from offset {skip}")

    total_embedded = 0

    while True:
        if _STOP:
            print(f"Stopped after embedding {total_embedded:,} chunks.")
            break
        if limit is not None and total_embedded >= limit:
            break

        fetch_size = batch_size
        if limit is not None:
            fetch_size = min(batch_size, limit - total_embedded)

        with neo4j_client.session() as session:
            result = session.run(CYPHER_FETCH_MISSING, skip=skip, limit=fetch_size)
            rows = [dict(r) for r in result]

        if not rows:
            print("No more chunks missing embeddings.")
            break

        texts = [r["text"] for r in rows]
        chunk_ids = [r["id"] for r in rows]

        try:
            embeddings = embed_texts_batch(texts, batch_size=batch_size, pause_between_batches=pause_between_batches)
        except RuntimeError as exc:
            logger.error("Embedding batch failed, stopping", extra={"error": str(exc), "skip": skip})
            _save_checkpoint(skip)
            raise

        write_rows = [{"chunk_id": cid, "embedding": emb} for cid, emb in zip(chunk_ids, embeddings)]
        neo4j_client.run_batch(CYPHER_WRITE_EMBEDDINGS, write_rows, batch_size=batch_size)

        total_embedded += len(rows)
        skip += len(rows)
        _save_checkpoint(skip)
        print(f"  Embedded {total_embedded:,} chunks (offset {skip:,})")

        if not _STOP and (limit is None or total_embedded < limit):
            time.sleep(pause_between_batches)

    print(f"\nEmbedding backfill complete: {total_embedded:,} chunks embedded.")


def main() -> None:
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(description="Backfill Gemini embeddings for QA dataset chunks")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to embed this run")
    parser.add_argument("--batch-size", type=int, default=settings.embed_batch_size, help="Embedding batch size")
    parser.add_argument("--pause", type=float, default=1.5, help="Seconds to pause between batches (rate-limit safety)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint, start from offset 0")
    args = parser.parse_args()

    embed_qa_chunks(
        limit=args.limit,
        batch_size=args.batch_size,
        pause_between_batches=args.pause,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
