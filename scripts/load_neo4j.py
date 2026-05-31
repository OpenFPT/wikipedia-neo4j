"""Load exported JSONL/CSV into Neo4j using batched UNWIND queries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.logging_utils import configure_logging, get_logger
from src.infrastructure.neo4j_client import neo4j_client

configure_logging(settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="load")
logger = get_logger(__name__)

CYPHER_PAGES = """
UNWIND $rows AS row
MERGE (p:Page {id: row.id})
SET p.title = row.title, p.url = row.url, p.summary = row.summary
"""

CYPHER_CHUNKS = """
UNWIND $rows AS row
MATCH (p:Page {id: row.page_id})
MERGE (c:Chunk {id: row.chunk_id})
SET c.text = row.text, c.sequence_number = row.seq, c.section = row.section
MERGE (p)-[:HAS_CHUNK]->(c)
"""

CYPHER_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (c:Chunk {id: row.chunk_id})
SET c.embedding = row.embedding
"""

CYPHER_ENTITIES = """
UNWIND $rows AS row
MERGE (e:Entity {id: row.entity_id})
SET e.name = row.name, e.aliases = row.aliases
"""

CYPHER_MENTIONS = """
UNWIND $rows AS row
MATCH (c:Chunk {id: row.chunk_id})
MATCH (e:Entity {id: row.entity_id})
MERGE (c)-[:MENTIONS]->(e)
"""

CYPHER_LINKS = """
UNWIND $rows AS row
MATCH (source:Page {title: row.source_title})
MATCH (target:Page {title: row.target_title})
MERGE (source)-[:LINKS_TO]->(target)
"""


def _read_jsonl(path: Path, limit: int | None = None):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            yield json.loads(line)


def _read_csv(path: Path, limit: int | None = None):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            yield row


def _load_in_batches(cypher: str, data_iter, batch_size: int, label: str) -> int:
    batch: list[dict] = []
    total = 0
    for row in data_iter:
        batch.append(row)
        if len(batch) >= batch_size:
            neo4j_client.run_batch(cypher, batch, batch_size=batch_size)
            total += len(batch)
            if total % (batch_size * 5) == 0:
                print(f"  {label}: {total:,} loaded")
            batch = []
    if batch:
        neo4j_client.run_batch(cypher, batch, batch_size=batch_size)
        total += len(batch)
    print(f"  {label}: {total:,} total")
    return total


def _clean_graph() -> None:
    """Delete all nodes and relationships in batches to avoid OOM."""
    print("Cleaning graph (deleting all nodes and relationships)...")
    total = 0
    while True:
        with neo4j_client.session() as session:
            result = session.run(
                "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS deleted"
            )
            deleted = result.single()["deleted"]
        if deleted == 0:
            break
        total += deleted
        print(f"  Deleted {total:,} nodes so far...")
    print(f"  Graph cleaned: {total:,} nodes removed")


def _drop_indexes() -> None:
    print("Dropping indexes for faster bulk load...")
    with neo4j_client.session() as session:
        result = session.run("SHOW INDEXES YIELD name RETURN name")
        names = [r["name"] for r in result]
        for name in names:
            if name.startswith("__"):
                continue
            try:
                session.run(f"DROP INDEX {name} IF EXISTS")
            except Exception:
                pass
    print(f"  Dropped {len(names)} indexes")


def _apply_entity_labels(input_dir: Path) -> None:
    """Set typed labels (Person, Organization, etc.) on entity nodes."""
    labels_seen: dict[str, list[str]] = {}
    for row in _read_jsonl(input_dir / "entities.jsonl"):
        label = row.get("label", "Entity")
        if label != "Entity":
            labels_seen.setdefault(label, []).append(row["entity_id"])

    for label, ids in labels_seen.items():
        cypher = f"""
        UNWIND $rows AS row
        MATCH (e:Entity {{id: row.entity_id}})
        SET e:{label}
        """
        batch_rows = [{"entity_id": eid} for eid in ids]
        neo4j_client.run_batch(cypher, batch_rows, batch_size=1000)
        print(f"  Applied :{label} to {len(ids):,} entities")


def load_neo4j(
    input_dir: str,
    limit: int | None,
    drop_indexes: bool,
    clean: bool,
    skip_embeddings: bool,
    page_batch: int,
    chunk_batch: int,
    entity_batch: int,
) -> None:
    inp = Path(input_dir)

    if clean:
        _clean_graph()

    if drop_indexes:
        _drop_indexes()

    print("Loading pages...")
    _load_in_batches(CYPHER_PAGES, _read_jsonl(inp / "pages.jsonl", limit), page_batch, "Pages")

    print("Loading chunks...")
    _load_in_batches(CYPHER_CHUNKS, _read_jsonl(inp / "chunks.jsonl"), chunk_batch, "Chunks")

    if not skip_embeddings and (inp / "chunk_embeddings.jsonl").exists():
        print("Loading embeddings...")
        _load_in_batches(CYPHER_EMBEDDINGS, _read_jsonl(inp / "chunk_embeddings.jsonl"), chunk_batch, "Embeddings")

    print("Loading entities...")
    _load_in_batches(CYPHER_ENTITIES, _read_jsonl(inp / "entities.jsonl"), entity_batch, "Entities")

    print("Applying entity labels...")
    _apply_entity_labels(inp)

    print("Loading mentions...")
    _load_in_batches(CYPHER_MENTIONS, _read_csv(inp / "mentions.csv"), entity_batch, "Mentions")

    if (inp / "links.csv").exists():
        print("Loading page links...")
        _load_in_batches(CYPHER_LINKS, _read_csv(inp / "links.csv"), entity_batch, "Links")

    if drop_indexes:
        print("Rebuilding schema (indexes + constraints)...")
        neo4j_client.setup_schema()

    print("\nLoad complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load exported JSONL/CSV into Neo4j")
    parser.add_argument("--input-dir", default="data/export", help="Directory with exported files")
    parser.add_argument("--limit", type=int, default=None, help="Limit pages to load (for testing)")
    parser.add_argument("--drop-indexes", action="store_true", help="Drop indexes before load, recreate after")
    parser.add_argument("--clean", action="store_true", help="Delete all nodes/relationships before loading")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip loading embeddings")
    parser.add_argument("--page-batch", type=int, default=settings.neo4j_page_batch, help="Page batch size")
    parser.add_argument("--chunk-batch", type=int, default=settings.neo4j_chunk_batch, help="Chunk batch size")
    parser.add_argument("--entity-batch", type=int, default=settings.neo4j_entity_batch, help="Entity batch size")
    args = parser.parse_args()

    load_neo4j(
        input_dir=args.input_dir,
        limit=args.limit,
        drop_indexes=args.drop_indexes,
        clean=args.clean,
        skip_embeddings=args.skip_embeddings,
        page_batch=args.page_batch,
        chunk_batch=args.chunk_batch,
        entity_batch=args.entity_batch,
    )


if __name__ == "__main__":
    main()
