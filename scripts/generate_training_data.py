"""Generate Text2Cypher training pairs from ViWiki-MHR dataset.

Produces JSONL with {"question": ..., "cypher": ..., "schema": ...} for
QLoRA fine-tuning of Vi-Qwen2-7B-RAG on Text2Cypher.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_utils import configure_logging, get_logger

configure_logging("INFO", log_dir="logs", task_name="gen_training")
logger = get_logger(__name__)

_STOP = False

GRAPH_SCHEMA = (
    "(:Page {title, url})-[:HAS_CHUNK]->(:Chunk {id, text, embedding}), "
    "(:Chunk)-[:MENTIONS]->(:Entity {name, type}), "
    "(:Page)-[:LINKS_TO]->(:Page), "
    "Entity labels: Person, Organization, Location, Work. "
    "Typed edges: MENTIONS_PERSON, MENTIONS_ORG, MENTIONS_LOCATION, MENTIONS_WORK."
)

# Required output aliases for all retrieval queries
RETURN_CLAUSE = (
    "RETURN p.title AS page_title, p.url AS page_url, "
    "c.id AS chunk_id, c.text AS chunk_text, 1.0 AS score LIMIT $top_k"
)


# --- Cypher Templates ---

TEMPLATES = [
    {
        "pattern": "find_entity",
        "cypher": (
            "MATCH (e:Entity) WHERE e.name CONTAINS $name "
            "MATCH (c:Chunk)-[:MENTIONS]->(e) "
            "MATCH (p:Page)-[:HAS_CHUNK]->(c) "
            f"{RETURN_CLAUSE}"
        ),
        "description": "Find chunks mentioning a specific entity",
    },
    {
        "pattern": "find_entity_typed",
        "cypher": (
            "MATCH (e:{entity_type}) WHERE e.name CONTAINS $name "
            "MATCH (c:Chunk)-[:MENTIONS]->(e) "
            "MATCH (p:Page)-[:HAS_CHUNK]->(c) "
            f"{RETURN_CLAUSE}"
        ),
        "description": "Find chunks mentioning a typed entity (Person/Org/Location/Work)",
    },
    {
        "pattern": "find_page",
        "cypher": (
            "MATCH (p:Page) WHERE p.title CONTAINS $title "
            "MATCH (p)-[:HAS_CHUNK]->(c:Chunk) "
            f"{RETURN_CLAUSE}"
        ),
        "description": "Find chunks from a specific page",
    },
    {
        "pattern": "multi_hop_link",
        "cypher": (
            "MATCH (p1:Page)-[:LINKS_TO]->(p2:Page)-[:HAS_CHUNK]->(c:Chunk) "
            "WHERE p1.title CONTAINS $source "
            "RETURN p2.title AS page_title, p2.url AS page_url, "
            "c.id AS chunk_id, c.text AS chunk_text, 1.0 AS score LIMIT $top_k"
        ),
        "description": "Follow page links to find related content",
    },
    {
        "pattern": "multi_hop_entity_bridge",
        "cypher": (
            "MATCH (p1:Page)-[:HAS_CHUNK]->(c1:Chunk)-[:MENTIONS]->(e:Entity)"
            "<-[:MENTIONS]-(c2:Chunk)<-[:HAS_CHUNK]-(p2:Page) "
            "WHERE p1.title CONTAINS $source AND p1 <> p2 "
            "RETURN p2.title AS page_title, p2.url AS page_url, "
            "c2.id AS chunk_id, c2.text AS chunk_text, 1.0 AS score LIMIT $top_k"
        ),
        "description": "Bridge two pages via shared entity mention",
    },
    {
        "pattern": "two_entity_intersection",
        "cypher": (
            "MATCH (e1:Entity {name: $entity1})<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:Entity {name: $entity2}) "
            "MATCH (p:Page)-[:HAS_CHUNK]->(c) "
            f"{RETURN_CLAUSE}"
        ),
        "description": "Find chunks mentioning two entities together",
    },
    {
        "pattern": "entity_from_page",
        "cypher": (
            "MATCH (p:Page {title: $title})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity) "
            "RETURN DISTINCT e.name AS entity_name, e.type AS entity_type, "
            "p.title AS page_title, p.url AS page_url, c.id AS chunk_id, "
            "c.text AS chunk_text, 1.0 AS score LIMIT $top_k"
        ),
        "description": "List entities mentioned in a page",
    },
    {
        "pattern": "linked_pages",
        "cypher": (
            "MATCH (p1:Page {title: $title})-[:LINKS_TO]->(p2:Page) "
            "MATCH (p2)-[:HAS_CHUNK]->(c:Chunk) "
            "RETURN p2.title AS page_title, p2.url AS page_url, "
            "c.id AS chunk_id, c.text AS chunk_text, 1.0 AS score LIMIT $top_k"
        ),
        "description": "Get content from pages linked by a given page",
    },
]

# Vietnamese question patterns mapped to templates
QUESTION_PATTERNS: list[dict] = [
    # Entity lookup
    {"regex": r"(?:thông tin|cho biết|mô tả).*(?:về|của)\s+(.+)", "template": "find_entity"},
    {"regex": r"(.+)\s+là (?:ai|gì|cái gì)", "template": "find_entity"},
    {"regex": r"(?:ai|người nào)\s+(?:là|đã)\s+(.+)", "template": "find_entity_typed"},
    # Page lookup
    {"regex": r"(?:bài viết|trang|nội dung).*(?:về|của)\s+(.+)", "template": "find_page"},
    {"regex": r"(?:tóm tắt|giới thiệu)\s+(.+)", "template": "find_page"},
    # Multi-hop
    {"regex": r"(.+)\s+(?:liên quan|có quan hệ|kết nối).*(?:với|đến)\s+(.+)", "template": "multi_hop_link"},
    {"regex": r"(?:mối quan hệ|liên hệ).*giữa\s+(.+)\s+và\s+(.+)", "template": "two_entity_intersection"},
    # Entity from page
    {"regex": r"(?:những|các)\s+(?:thực thể|đối tượng|nhân vật).*(?:trong|của)\s+(.+)", "template": "entity_from_page"},
    # Linked pages
    {"regex": r"(?:những|các)\s+(?:trang|bài).*(?:liên kết|link).*(?:từ|của)\s+(.+)", "template": "linked_pages"},
]


def _handle_sigint(sig, frame):
    global _STOP
    _STOP = True
    print("\nGraceful shutdown requested, finishing current batch...")


def _extract_entities_from_question(question: str) -> list[str]:
    """Extract potential entity names from a Vietnamese question using heuristics."""
    # Remove common question words
    cleaned = re.sub(
        r"^(ai|gì|nào|ở đâu|khi nào|bao giờ|tại sao|vì sao|như thế nào|"
        r"cho biết|hãy|mô tả|giới thiệu|tóm tắt|liệt kê)\s*",
        "",
        question.strip(),
        flags=re.IGNORECASE,
    )
    # Look for capitalized sequences (likely proper nouns in Vietnamese)
    entities = re.findall(r"[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*", cleaned)
    if not entities:
        # Fallback: extract quoted strings
        entities = re.findall(r'"([^"]+)"', question)
    if not entities:
        # Fallback: take the longest noun-like phrase
        words = cleaned.split()
        if len(words) >= 2:
            entities = [" ".join(words[:3])]
        elif words:
            entities = [words[0]]
    return entities


def _match_question_to_template(question: str) -> tuple[str, dict[str, str]] | None:
    """Match a question to a Cypher template and extract parameters."""
    for pattern_info in QUESTION_PATTERNS:
        match = re.search(pattern_info["regex"], question, re.IGNORECASE)
        if match:
            groups = match.groups()
            template_name = pattern_info["template"]
            params: dict[str, str] = {}

            if template_name in ("find_entity", "find_entity_typed"):
                params["name"] = groups[0].strip()
            elif template_name == "find_page":
                params["title"] = groups[0].strip()
            elif template_name == "multi_hop_link":
                params["source"] = groups[0].strip()
            elif template_name == "two_entity_intersection":
                params["entity1"] = groups[0].strip()
                params["entity2"] = groups[1].strip()
            elif template_name == "entity_from_page":
                params["title"] = groups[0].strip()
            elif template_name == "linked_pages":
                params["title"] = groups[0].strip()

            return template_name, params

    return None


def _generate_cypher_from_template(template_name: str, params: dict[str, str]) -> str:
    """Generate a concrete Cypher query from a template and parameters."""
    template = next((t for t in TEMPLATES if t["pattern"] == template_name), None)
    if template is None:
        raise ValueError(f"Unknown template: {template_name}")

    cypher = template["cypher"]

    # Replace typed entity placeholder if needed
    if "{entity_type}" in cypher:
        entity_type = _guess_entity_type(params.get("name", ""))
        cypher = cypher.replace("{entity_type}", entity_type)

    return cypher


def _guess_entity_type(name: str) -> str:
    """Guess entity type from name heuristics."""
    # Common Vietnamese location indicators
    location_indicators = ["tỉnh", "thành phố", "huyện", "quận", "xã", "sông", "núi", "biển"]
    org_indicators = ["đảng", "công ty", "tổ chức", "trường", "đại học", "viện", "bộ"]

    name_lower = name.lower()
    for indicator in location_indicators:
        if indicator in name_lower:
            return "Location"
    for indicator in org_indicators:
        if indicator in name_lower:
            return "Organization"
    return "Entity"


def _generate_from_heuristic(question: str, entities: list[str]) -> tuple[str, str] | None:
    """Generate a Cypher query using heuristic matching when regex fails."""
    if not entities:
        return None

    # Default: entity lookup for the first entity found
    entity = entities[0]

    # Check if question implies multi-hop
    multi_hop_keywords = ["liên quan", "quan hệ", "kết nối", "ảnh hưởng", "tác động"]
    if any(kw in question.lower() for kw in multi_hop_keywords) and len(entities) >= 2:
        template_name = "two_entity_intersection"
        cypher = _generate_cypher_from_template(
            template_name, {"entity1": entities[0], "entity2": entities[1]}
        )
    elif any(kw in question.lower() for kw in multi_hop_keywords):
        template_name = "multi_hop_entity_bridge"
        cypher = _generate_cypher_from_template(template_name, {"source": entity})
    else:
        template_name = "find_entity"
        cypher = _generate_cypher_from_template(template_name, {"name": entity})

    return template_name, cypher


def generate_training_pair(sample: dict) -> dict | None:
    """Generate a single training pair from a QA sample.

    Args:
        sample: dict with at least 'question' field. May also have
                'reasoning_type', 'kg_path', 'entities'.

    Returns:
        dict with 'question', 'cypher', 'schema' or None if cannot generate.
    """
    question = sample.get("question", "").strip()
    if not question:
        return None

    # Strategy 1: Match question to template via regex
    match_result = _match_question_to_template(question)
    if match_result:
        template_name, params = match_result
        cypher = _generate_cypher_from_template(template_name, params)
        return {"question": question, "cypher": cypher, "schema": GRAPH_SCHEMA}

    # Strategy 2: Use entities from the sample metadata
    entities = sample.get("entities", {})
    if isinstance(entities, dict):
        entity_names = list(entities.values())
    elif isinstance(entities, list):
        entity_names = entities
    else:
        entity_names = []

    # Strategy 3: Extract entities from question text
    if not entity_names:
        entity_names = _extract_entities_from_question(question)

    result = _generate_from_heuristic(question, entity_names)
    if result:
        _, cypher = result
        return {"question": question, "cypher": cypher, "schema": GRAPH_SCHEMA}

    return None


def generate_training_data(
    input_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> dict:
    """Process input JSONL and generate training pairs.

    Args:
        input_path: Path to input JSONL (viwiki_mhr.jsonl or similar).
        output_path: Path for output JSONL.
        limit: Maximum number of pairs to generate.

    Returns:
        Stats dict with total, generated, skipped counts.
    """
    stats = {"total": 0, "generated": 0, "skipped": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if _STOP:
                logger.info("Interrupted by user signal")
                break

            line = line.strip()
            if not line:
                continue

            stats["total"] += 1
            sample = json.loads(line)
            pair = generate_training_pair(sample)

            if pair:
                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                stats["generated"] += 1
            else:
                stats["skipped"] += 1

            if limit and stats["generated"] >= limit:
                logger.info("Reached generation limit", extra={"limit": limit})
                break

            if stats["total"] % 1000 == 0:
                logger.info(
                    "Progress",
                    extra={"total": stats["total"], "generated": stats["generated"]},
                )

    logger.info("Training data generation complete", extra=stats)
    return stats


def main() -> None:
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(
        description="Generate Text2Cypher training pairs from ViWiki-MHR dataset"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/viwiki_mhr.jsonl"),
        help="Input JSONL with multi-hop QA samples (default: data/viwiki_mhr.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/text2cypher_train.jsonl"),
        help="Output JSONL path (default: data/training/text2cypher_train.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of training pairs to generate",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found", extra={"path": str(args.input)})
        sys.exit(1)

    stats = generate_training_data(args.input, args.output, limit=args.limit)
    print(
        f"Done: {stats['generated']}/{stats['total']} pairs generated, "
        f"{stats['skipped']} skipped. Output: {args.output}"
    )


if __name__ == "__main__":
    main()
