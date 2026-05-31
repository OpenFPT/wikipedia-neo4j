"""Verify Neo4j ingestion integrity after loading exported data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infrastructure.neo4j_client import neo4j_client


def _query_single(cypher: str) -> int:
    with neo4j_client.session() as session:
        result = session.run(cypher)
        return result.single()[0]


def verify(input_dir: str = "data/export") -> None:
    inp = Path(input_dir)
    stats_file = inp / "stats.json"
    expected = json.loads(stats_file.read_text()) if stats_file.exists() else {}

    print("=== Neo4j Ingestion Verification ===\n")

    page_count = _query_single("MATCH (p:Page) RETURN count(p)")
    chunk_count = _query_single("MATCH (c:Chunk) RETURN count(c)")
    entity_count = _query_single("MATCH (e:Entity) RETURN count(e)")
    mention_count = _query_single("MATCH ()-[r:MENTIONS]->() RETURN count(r)")

    print(f"Pages:    {page_count:>10,}  (expected: {expected.get('pages', '?')})")
    print(f"Chunks:   {chunk_count:>10,}  (expected: {expected.get('chunks', '?')})")
    print(f"Entities: {entity_count:>10,}  (expected: {expected.get('entities', '?')})")
    print(f"Mentions: {mention_count:>10,}  (expected: {expected.get('mentions', '?')})")

    orphan_chunks = _query_single(
        "MATCH (c:Chunk) WHERE NOT EXISTS { MATCH (p:Page)-[:HAS_CHUNK]->(c) } RETURN count(c)"
    )
    orphan_entities = _query_single(
        "MATCH (e:Entity) WHERE NOT EXISTS { MATCH ()-[:MENTIONS]->(e) } RETURN count(e)"
    )
    chunks_with_embedding = _query_single(
        "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c)"
    )

    print(f"\nOrphan chunks (no page):     {orphan_chunks}")
    print(f"Orphan entities (no mention): {orphan_entities}")
    print(f"Chunks with embeddings:       {chunks_with_embedding:,} / {chunk_count:,}")

    if chunk_count > 0:
        pct = chunks_with_embedding / chunk_count * 100
        print(f"Embedding coverage:           {pct:.1f}%")

    issues = []
    if orphan_chunks > 0:
        issues.append(f"{orphan_chunks} orphan chunks")
    if expected.get("pages") and page_count < expected["pages"] * 0.95:
        issues.append(f"Page count {page_count} is <95% of expected {expected['pages']}")

    if issues:
        print(f"\nIssues found: {', '.join(issues)}")
    else:
        print("\nAll checks passed")


if __name__ == "__main__":
    verify()
