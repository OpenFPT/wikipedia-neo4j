"""Backfill entities for pages already ingested into Neo4j."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uuid

from src.config import settings
from src.logging_utils import get_logger
from src.infrastructure.neo4j_client import Neo4jClient
from src.extraction.ner import extract_entities

logger = get_logger(__name__)


def backfill_entities(batch_size: int = 50) -> dict:
    """Extract entities from existing chunks and write to Neo4j.

    Returns stats dict with pages_processed, entities_created, mentions_created.
    """
    client = Neo4jClient()
    stats = {"pages_processed": 0, "entities_created": 0, "mentions_created": 0}

    with client.driver.session() as session:
        pages = session.run(
            "MATCH (p:Page) RETURN p.id AS id, p.title AS title ORDER BY p.title"
        )
        page_list = [(r["id"], r["title"]) for r in pages]

    total = len(page_list)
    logger.info(f"Backfilling entities for {total} pages (NER backend: {settings.ner_backend})")

    for i in range(0, total, batch_size):
        batch = page_list[i : i + batch_size]

        for page_id, title in batch:
            with client.driver.session() as session:
                chunks = session.run(
                    "MATCH (p:Page {id: $pid})-[:HAS_CHUNK]->(c:Chunk) "
                    "RETURN c.id AS chunk_id, c.text AS text ORDER BY c.sequence_number",
                    pid=page_id,
                )
                chunk_list = [(r["chunk_id"], r["text"]) for r in chunks]

            full_text = " ".join(text for _, text in chunk_list)
            entities = extract_entities(full_text)

            if not entities:
                stats["pages_processed"] += 1
                print(f"  [{stats['pages_processed']}/{total}] {title}: 0 entities", flush=True)
                continue

            with client.driver.session() as session:
                for name, entity_type in entities:
                    entity_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name.lower()))
                    label = entity_type if entity_type in ("Person", "Organization", "Location", "Work") else "Entity"

                    session.run(
                        f"MERGE (e:{label} {{id: $entity_id}}) SET e.name = $name, e.type = $type",
                        entity_id=entity_id,
                        name=name,
                        type=entity_type,
                    )
                    stats["entities_created"] += 1

                    for chunk_id, chunk_text in chunk_list:
                        if name.lower() in chunk_text.lower():
                            session.run(
                                "MATCH (c:Chunk {id: $cid}) "
                                "MATCH (e {id: $eid}) "
                                "MERGE (c)-[:MENTIONS]->(e)",
                                cid=chunk_id,
                                eid=entity_id,
                            )
                            stats["mentions_created"] += 1

            stats["pages_processed"] += 1
            print(f"  [{stats['pages_processed']}/{total}] {title}: {len(entities)} entities", flush=True)

    client.close()
    return stats


if __name__ == "__main__":
    import sys

    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    result = backfill_entities(batch_size=batch)
    print(f"\nDone: {result['pages_processed']} pages, "
          f"{result['entities_created']} entities, "
          f"{result['mentions_created']} mentions")
