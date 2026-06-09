"""Migrate Neo4j graph schema: consolidate Entity labels, drop redundant edges.

Steps:
1. Remove generic :Entity label from nodes that already have a typed label
2. Assign 'Unknown' entities a best-effort type or remove if noise
3. Drop redundant MENTIONS_X edges (keep only generic MENTIONS)
4. Add vector index for Chunk embeddings
5. Add Event label support

Usage:
    uv run python -m scripts.migrate_schema
"""

from __future__ import annotations

import argparse

from src.logging_utils import get_logger
from src.infrastructure.neo4j_client import neo4j_client

logger = get_logger(__name__)


def step1_consolidate_typed_entities() -> dict[str, int]:
    """Remove :Entity label from nodes that already have Person/Org/Location/Work."""
    stats = {}
    with neo4j_client.session() as s:
        for label in ["Person", "Organization", "Location", "Work"]:
            r = s.run(
                f"MATCH (e:Entity:{label}) REMOVE e:Entity RETURN count(e) AS c"
            )
            count = r.single()["c"]
            stats[label] = count
            logger.info(f"Removed :Entity from :{label} nodes", extra={"count": count})
    return stats


def step2_clean_untyped_entities() -> dict[str, int]:
    """Remove noise entities: single char, numeric, or very short names."""
    with neo4j_client.session() as s:
        r = s.run("""
            MATCH (e:Entity)
            WHERE size(e.name) < 2
            DETACH DELETE e
            RETURN count(e) AS c
        """)
        deleted_short = r.single()["c"]

        r = s.run("""
            MATCH (e:Entity)
            WHERE e.name =~ '^[0-9,.%\\s]+$'
            DETACH DELETE e
            RETURN count(e) AS c
        """)
        deleted_numeric = r.single()["c"]

        r = s.run("MATCH (e:Entity) WHERE e.type IS NULL OR e.type = 'Unknown' RETURN count(e) AS c")
        remaining = r.single()["c"]

    stats = {"deleted_short": deleted_short, "deleted_numeric": deleted_numeric, "remaining_untyped": remaining}
    logger.info("Cleaned untyped entities", extra=stats)
    return stats


def step3_drop_redundant_mention_edges() -> dict[str, int]:
    """Drop typed MENTIONS_X edges — generic MENTIONS is sufficient with typed labels."""
    stats = {}
    with neo4j_client.session() as s:
        for rel_type in ["MENTIONS_PERSON", "MENTIONS_ORG", "MENTIONS_LOCATION", "MENTIONS_WORK"]:
            r = s.run(f"MATCH ()-[r:{rel_type}]->() DELETE r RETURN count(r) AS c")
            count = r.single()["c"]
            stats[rel_type] = count
            logger.info(f"Dropped {rel_type} edges", extra={"count": count})
    return stats


def step4_add_vector_index() -> None:
    """Create native Neo4j vector index on Chunk embeddings."""
    with neo4j_client.session() as s:
        r = s.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN size(c.embedding) AS dim LIMIT 1")
        rec = r.single()
        if not rec:
            logger.warning("No embeddings found, skipping vector index")
            return

        dim = rec["dim"]
        logger.info(f"Detected embedding dimension: {dim}")

        s.run(f"""
            CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)
        logger.info("Vector index created", extra={"dimensions": dim})


def step5_add_event_support() -> None:
    """Add Event label index."""
    with neo4j_client.session() as s:
        s.run("CREATE INDEX event_name_idx IF NOT EXISTS FOR (e:Event) ON (e.name)")
        logger.info("Event index created")


def run_migration(skip_vector: bool = False) -> None:
    """Run all migration steps."""
    print("Step 1: Consolidate typed entities...")
    s1 = step1_consolidate_typed_entities()
    print(f"  {s1}")

    print("Step 2: Clean noise entities...")
    s2 = step2_clean_untyped_entities()
    print(f"  {s2}")

    print("Step 3: Drop redundant MENTIONS_X edges...")
    s3 = step3_drop_redundant_mention_edges()
    print(f"  {s3}")

    if not skip_vector:
        print("Step 4: Add vector index...")
        step4_add_vector_index()
        print("  Done")

    print("Step 5: Add Event label support...")
    step5_add_event_support()
    print("  Done")

    with neo4j_client.session() as s:
        print("\n=== Final Graph State ===")
        for label in ["Page", "Chunk", "Entity", "Person", "Organization", "Location", "Work", "Event"]:
            r = s.run(f"MATCH (n:{label}) RETURN count(n) AS c")
            print(f"  {label}: {r.single()['c']}")
        print()
        r = s.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel, count(r) AS cnt
            ORDER BY cnt DESC
        """)
        print("  Relationships:")
        for rec in r:
            print(f"    {rec['rel']}: {rec['cnt']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Neo4j graph schema")
    parser.add_argument("--skip-vector", action="store_true", help="Skip vector index creation")
    args = parser.parse_args()
    run_migration(skip_vector=args.skip_vector)


if __name__ == "__main__":
    main()
