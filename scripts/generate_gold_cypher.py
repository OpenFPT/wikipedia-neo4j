"""Generate gold Cypher queries for multi-hop QA samples.

Given a multi-hop question and its KG path, produce the corresponding
Cypher query that would retrieve the answer from Neo4j.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.logging_utils import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "viwiki_mhr"


@dataclass
class CypherTemplate:
    """A template for generating Cypher from a KG path pattern."""

    pattern: str
    cypher: str
    description: str


TEMPLATES: list[CypherTemplate] = [
    CypherTemplate(
        pattern="bridge_2hop",
        cypher=(
            "MATCH (e1:{label1} {{name: $entity1}})-[:{rel1}]->(e2:{label2})"
            "-[:{rel2}]->(e3:{label3}) "
            "RETURN e3.name AS answer"
        ),
        description="2-hop bridge: A->B->C, answer is C",
    ),
    CypherTemplate(
        pattern="bridge_3hop",
        cypher=(
            "MATCH (e1:{label1} {{name: $entity1}})-[:{rel1}]->(e2:{label2})"
            "-[:{rel2}]->(e3:{label3})-[:{rel3}]->(e4:{label4}) "
            "RETURN e4.name AS answer"
        ),
        description="3-hop bridge: A->B->C->D, answer is D",
    ),
    CypherTemplate(
        pattern="comparison",
        cypher=(
            "MATCH (e1:{label1} {{name: $entity1}})-[:{rel1}]->(shared:{shared_label})"
            "<-[:{rel2}]-(e2:{label2} {{name: $entity2}}) "
            "RETURN shared.name AS answer"
        ),
        description="Comparison: two entities sharing a common neighbor",
    ),
    CypherTemplate(
        pattern="intersection",
        cypher=(
            "MATCH (e1:{label1} {{name: $entity1}})-[:{rel1}]->(target:{target_label}), "
            "(e2:{label2} {{name: $entity2}})-[:{rel2}]->(target) "
            "RETURN target.name AS answer"
        ),
        description="Intersection: two paths converging on same target",
    ),
    CypherTemplate(
        pattern="temporal",
        cypher=(
            "MATCH (e1:{label1} {{name: $entity1}})-[:{rel1}]->(ev:Event) "
            "WHERE ev.year {operator} $year "
            "RETURN ev.name AS answer, ev.year AS year "
            "ORDER BY ev.year {order} LIMIT 1"
        ),
        description="Temporal: filter events by time constraint",
    ),
]


def generate_cypher_for_sample(sample: dict) -> str | None:
    """Generate a gold Cypher query for a single multi-hop QA sample.

    Args:
        sample: dict with keys:
            - reasoning_type: bridge_2hop, bridge_3hop, comparison, intersection, temporal
            - kg_path: list of dicts with entity, label, relationship info
            - entities: dict mapping entity placeholders to actual names

    Returns:
        Formatted Cypher query string, or None if pattern not recognized.
    """
    reasoning_type = sample.get("reasoning_type", "")
    kg_path = sample.get("kg_path", [])
    entities = sample.get("entities", {})

    template = next((t for t in TEMPLATES if t.pattern == reasoning_type), None)
    if template is None:
        logger.warning("Unknown reasoning type", extra={"type": reasoning_type})
        return None

    cypher = template.cypher
    for key, value in entities.items():
        cypher = cypher.replace(f"{{{key}}}", value)

    # Pass 1: Fill labels and relationships from each hop's own label
    for i, hop in enumerate(kg_path):
        cypher = cypher.replace(f"{{label{i+1}}}", hop.get("label", "Unknown"))
        cypher = cypher.replace(f"{{rel{i+1}}}", hop.get("relationship", "RELATED_TO"))

    # Pass 2: Fill remaining label placeholders from target_label (for bridge end-nodes)
    remaining = re.findall(r"\{label\d+\}", cypher)
    if remaining and kg_path:
        for i, hop in enumerate(kg_path):
            target = hop.get("target_label")
            if target:
                placeholder = f"{{label{i+2}}}"
                if placeholder in remaining:
                    cypher = cypher.replace(placeholder, target)

    # Fill any still-unfilled label placeholders with last known target
    remaining = re.findall(r"\{label\d+\}", cypher)
    if remaining and kg_path:
        fallback = kg_path[-1].get("target_label", kg_path[-1].get("label", "Unknown"))
        for placeholder in remaining:
            cypher = cypher.replace(placeholder, fallback)

    # Handle special template vars
    if "{shared_label}" in cypher:
        cypher = cypher.replace("{shared_label}", kg_path[-1].get("target_label", kg_path[-1].get("label", "Unknown")))
    if "{target_label}" in cypher:
        cypher = cypher.replace("{target_label}", kg_path[-1].get("target_label", kg_path[-1].get("label", "Unknown")))

    return cypher


def process_dataset(input_path: Path, output_path: Path) -> dict:
    """Add cypher_query field to each sample in a JSONL dataset."""
    stats = {"total": 0, "generated": 0, "skipped": 0}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            sample = json.loads(line)

            cypher = generate_cypher_for_sample(sample)
            if cypher:
                sample["cypher_query"] = cypher
                stats["generated"] += 1
            else:
                sample["cypher_query"] = None
                stats["skipped"] += 1

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Cypher generation complete", extra=stats)
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate gold Cypher queries")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with multi-hop samples")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL with cypher_query field")
    args = parser.parse_args()

    stats = process_dataset(args.input, args.output)
    print(f"Done: {stats['generated']}/{stats['total']} queries generated, {stats['skipped']} skipped")


if __name__ == "__main__":
    main()
