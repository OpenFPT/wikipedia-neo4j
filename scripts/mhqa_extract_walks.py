"""Stage 1: Extract bridge walk structures from Neo4j for MHQA generation."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.infrastructure.neo4j_client import neo4j_client
from src.logging_utils import configure_logging, get_logger

configure_logging(settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="mhqa_walks")
logger = get_logger(__name__)

_STOP = False


def _handle_sigint(sig, frame):
    global _STOP
    _STOP = True
    print("\nGraceful shutdown requested, finishing current batch...")


# -- Cypher queries for walk extraction --

_QUERY_2HOP_BRIDGE = """
MATCH (p1:Page)-[:HAS_CHUNK]->(c1:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(c2:Chunk)<-[:HAS_CHUNK]-(p2:Page)
WHERE p1 <> p2
  AND id(p1) < id(p2)
  AND size(c1.text) >= 200
  AND size(c2.text) >= 200
WITH p1, p2, e, c1, c2
LIMIT $limit
RETURN p1.id AS page1_id, p1.title AS page1_title,
       p2.id AS page2_id, p2.title AS page2_title,
       e.name AS entity_name, labels(e) AS entity_labels,
       c1.id AS chunk1_id, c1.text AS chunk1_text,
       c2.id AS chunk2_id, c2.text AS chunk2_text
"""

_QUERY_3HOP_BRIDGE = """
MATCH (p1:Page)-[:HAS_CHUNK]->(c1:Chunk)-[:MENTIONS]->(e1:Entity)-[:RELATED_TO|LINKS_TO*1..2]-(e2:Entity)<-[:MENTIONS]-(c2:Chunk)<-[:HAS_CHUNK]-(p2:Page)
WHERE p1 <> p2
  AND e1 <> e2
  AND id(p1) < id(p2)
  AND size(c1.text) >= 200
  AND size(c2.text) >= 200
WITH p1, p2, e1, e2, c1, c2
LIMIT $limit
RETURN p1.id AS page1_id, p1.title AS page1_title,
       p2.id AS page2_id, p2.title AS page2_title,
       e1.name AS entity1_name, labels(e1) AS entity1_labels,
       e2.name AS entity2_name, labels(e2) AS entity2_labels,
       c1.id AS chunk1_id, c1.text AS chunk1_text,
       c2.id AS chunk2_id, c2.text AS chunk2_text
"""

_QUERY_COMPARISON = """
MATCH (p1:Page)-[:HAS_CHUNK]->(c1:Chunk)-[:MENTIONS]->(e1:Entity),
      (p2:Page)-[:HAS_CHUNK]->(c2:Chunk)-[:MENTIONS]->(e2:Entity)
WHERE p1 <> p2
  AND id(p1) < id(p2)
  AND any(lbl IN labels(e1) WHERE lbl IN labels(e2) AND lbl <> 'Entity')
  AND size(c1.text) >= 200
  AND size(c2.text) >= 200
WITH p1, p2, e1, e2, c1, c2,
     [lbl IN labels(e1) WHERE lbl IN labels(e2) AND lbl <> 'Entity'][0] AS shared_type
LIMIT $limit
RETURN p1.id AS page1_id, p1.title AS page1_title,
       p2.id AS page2_id, p2.title AS page2_title,
       e1.name AS entity1_name, e2.name AS entity2_name,
       shared_type,
       c1.id AS chunk1_id, c1.text AS chunk1_text,
       c2.id AS chunk2_id, c2.text AS chunk2_text
"""


def _split_sentences(text: str) -> list[str]:
    """Split chunk text into sentences (simple Vietnamese-aware split)."""
    import re

    # Split on period/question/exclamation followed by space+uppercase or end
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _entity_in_text(entity_name: str, text: str) -> bool:
    """Check if entity name appears in text (case-insensitive)."""
    return entity_name.lower() in text.lower()


def _extract_entity_type(labels: list[str]) -> str:
    """Get the most specific entity type from Neo4j labels."""
    priority = ["Person", "Organization", "Location", "Work"]
    for t in priority:
        if t in labels:
            return t
    return "Entity"


def _build_walk_record(
    walk_id: str,
    walk_type: str,
    num_hops: int,
    page1: dict,
    page2: dict,
    bridge_entities: list[str],
    entity_types: list[str],
) -> dict:
    """Construct a walk record matching the spec schema."""
    return {
        "walk_id": walk_id,
        "type": walk_type,
        "num_hops": num_hops,
        "pages": [page1, page2],
        "bridge_entities": bridge_entities,
        "entity_types": entity_types,
    }


def extract_2hop_bridges(limit: int) -> list[dict]:
    """Extract 2-hop bridge walks from Neo4j."""
    walks: list[dict] = []
    seen: set[tuple] = set()

    with neo4j_client.session() as session:
        result = session.run(_QUERY_2HOP_BRIDGE, limit=limit)
        for record in result:
            key = (record["page1_id"], record["page2_id"], record["entity_name"])
            if key in seen:
                continue
            seen.add(key)

            entity_name = record["entity_name"]
            chunk1_text = record["chunk1_text"]
            chunk2_text = record["chunk2_text"]

            # Verify bridge entity appears in both chunks
            if not _entity_in_text(entity_name, chunk1_text):
                continue
            if not _entity_in_text(entity_name, chunk2_text):
                continue

            sentences1 = _split_sentences(chunk1_text)
            sentences2 = _split_sentences(chunk2_text)

            # Skip if too few sentences
            if len(sentences1) < 3 or len(sentences2) < 3:
                continue

            entity_type = _extract_entity_type(record["entity_labels"])

            page1 = {
                "page_id": record["page1_id"],
                "title": record["page1_title"],
                "chunks": [{"chunk_id": record["chunk1_id"], "text": chunk1_text, "sentences": sentences1}],
            }
            page2 = {
                "page_id": record["page2_id"],
                "title": record["page2_title"],
                "chunks": [{"chunk_id": record["chunk2_id"], "text": chunk2_text, "sentences": sentences2}],
            }

            walk_id = f"w-{len(walks):05d}"
            walks.append(
                _build_walk_record(walk_id, "bridge", 2, page1, page2, [entity_name], [entity_type])
            )

            if _STOP:
                break

    logger.info(f"Extracted {len(walks)} 2-hop bridge walks (from {len(seen)} candidates)")
    return walks


def extract_3hop_bridges(limit: int, id_offset: int = 0) -> list[dict]:
    """Extract 3-hop bridge walks from Neo4j."""
    walks: list[dict] = []
    seen: set[tuple] = set()

    with neo4j_client.session() as session:
        result = session.run(_QUERY_3HOP_BRIDGE, limit=limit)
        for record in result:
            key = (record["page1_id"], record["page2_id"], record["entity1_name"], record["entity2_name"])
            if key in seen:
                continue
            seen.add(key)

            e1_name = record["entity1_name"]
            e2_name = record["entity2_name"]
            chunk1_text = record["chunk1_text"]
            chunk2_text = record["chunk2_text"]

            # Verify entities appear in respective chunks
            if not _entity_in_text(e1_name, chunk1_text):
                continue
            if not _entity_in_text(e2_name, chunk2_text):
                continue

            sentences1 = _split_sentences(chunk1_text)
            sentences2 = _split_sentences(chunk2_text)

            if len(sentences1) < 3 or len(sentences2) < 3:
                continue

            e1_type = _extract_entity_type(record["entity1_labels"])
            e2_type = _extract_entity_type(record["entity2_labels"])

            page1 = {
                "page_id": record["page1_id"],
                "title": record["page1_title"],
                "chunks": [{"chunk_id": record["chunk1_id"], "text": chunk1_text, "sentences": sentences1}],
            }
            page2 = {
                "page_id": record["page2_id"],
                "title": record["page2_title"],
                "chunks": [{"chunk_id": record["chunk2_id"], "text": chunk2_text, "sentences": sentences2}],
            }

            walk_id = f"w-{id_offset + len(walks):05d}"
            walks.append(
                _build_walk_record(walk_id, "bridge", 3, page1, page2, [e1_name, e2_name], [e1_type, e2_type])
            )

            if _STOP:
                break

    logger.info(f"Extracted {len(walks)} 3-hop bridge walks (from {len(seen)} candidates)")
    return walks


def extract_comparisons(limit: int, id_offset: int = 0) -> list[dict]:
    """Extract comparison walks from Neo4j."""
    walks: list[dict] = []
    seen: set[tuple] = set()

    with neo4j_client.session() as session:
        result = session.run(_QUERY_COMPARISON, limit=limit)
        for record in result:
            key = (record["page1_id"], record["page2_id"])
            if key in seen:
                continue
            seen.add(key)

            chunk1_text = record["chunk1_text"]
            chunk2_text = record["chunk2_text"]

            sentences1 = _split_sentences(chunk1_text)
            sentences2 = _split_sentences(chunk2_text)

            if len(sentences1) < 3 or len(sentences2) < 3:
                continue

            page1 = {
                "page_id": record["page1_id"],
                "title": record["page1_title"],
                "chunks": [{"chunk_id": record["chunk1_id"], "text": chunk1_text, "sentences": sentences1}],
            }
            page2 = {
                "page_id": record["page2_id"],
                "title": record["page2_title"],
                "chunks": [{"chunk_id": record["chunk2_id"], "text": chunk2_text, "sentences": sentences2}],
            }

            walk_id = f"w-{id_offset + len(walks):05d}"
            walks.append(
                _build_walk_record(
                    walk_id,
                    "comparison",
                    2,
                    page1,
                    page2,
                    [record["entity1_name"], record["entity2_name"]],
                    [record["shared_type"]],
                )
            )

            if _STOP:
                break

    logger.info(f"Extracted {len(walks)} comparison walks (from {len(seen)} candidates)")
    return walks


def main():
    parser = argparse.ArgumentParser(description="Extract bridge walks from Neo4j for MHQA generation")
    parser.add_argument("--limit", type=int, default=5000, help="Max walks to extract total")
    parser.add_argument("--output", type=str, default="data/mhqa/walks.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Distribute limit across walk types per spec: 60% 2-hop, 30% 3-hop, 10% comparison
    limit_2hop = int(args.limit * 0.6)
    limit_3hop = int(args.limit * 0.3)
    limit_comp = args.limit - limit_2hop - limit_3hop

    logger.info(f"Extracting walks: 2-hop={limit_2hop}, 3-hop={limit_3hop}, comparison={limit_comp}")

    walks_2hop = extract_2hop_bridges(limit_2hop)
    if _STOP:
        logger.warning("Interrupted during 2-hop extraction")

    walks_3hop = []
    if not _STOP:
        walks_3hop = extract_3hop_bridges(limit_3hop, id_offset=len(walks_2hop))

    walks_comp = []
    if not _STOP:
        walks_comp = extract_comparisons(limit_comp, id_offset=len(walks_2hop) + len(walks_3hop))

    all_walks = walks_2hop + walks_3hop + walks_comp

    # Re-number walk IDs sequentially
    for i, walk in enumerate(all_walks):
        walk["walk_id"] = f"w-{i:05d}"

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for walk in all_walks:
            f.write(json.dumps(walk, ensure_ascii=False) + "\n")

    stats = {
        "total": len(all_walks),
        "2hop_bridge": len(walks_2hop),
        "3hop_bridge": len(walks_3hop),
        "comparison": len(walks_comp),
    }
    logger.info(f"Wrote {len(all_walks)} walks to {output_path}")
    logger.info(f"Stats: {stats}")

    # Write stats alongside output
    stats_path = output_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
