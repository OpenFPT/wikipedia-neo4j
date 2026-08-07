"""Ingest multi-hop QA datasets (seeds/ViQuAD2/vi-exclusive) into Neo4j.

Builds Page/Chunk/Entity nodes from embedded Wikipedia context segments, plus
a Question layer (SUPPORTED_BY -> Chunk, BRIDGES -> Page) capturing the
multi-hop benchmark structure. Single-pass, in-process, batched UNWIND writes
via neo4j_client.run_batch — sized for ~8-9k QA records, not the 1.6M-article
bulk pipeline scripts/load_neo4j.py was built for.

Usage:
    uv run python -m scripts.ingest_qa_datasets --dry-run --limit 20
    uv run python -m scripts.ingest_qa_datasets
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from src.config import settings
from src.extraction.ner import extract_entities
from src.infrastructure.neo4j_client import neo4j_client
from src.logging_utils import configure_logging, get_logger

configure_logging(settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="ingest_qa")
logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent

CYPHER_PAGES = """
UNWIND $rows AS row
MERGE (p:Page {id: row.id})
SET p.title = row.title, p.url = row.url, p.summary = row.summary
"""

CYPHER_CHUNKS = """
UNWIND $rows AS row
MATCH (p:Page {id: row.page_id})
MERGE (c:Chunk {id: row.id})
SET c.text = row.text,
    c.sequence_number = row.seq,
    c.section = row.section,
    c.source_title = row.source_title,
    c.source_segment_id = row.source_segment_id
MERGE (p)-[:HAS_CHUNK]->(c)
"""

CYPHER_ENTITIES = """
UNWIND $rows AS row
MERGE (e:Entity {id: row.id})
SET e.name = row.name
"""

CYPHER_ENTITY_LABEL = """
UNWIND $rows AS row
MATCH (e:Entity {id: row.entity_id})
SET e:%s
"""

CYPHER_MENTIONS = """
UNWIND $rows AS row
MATCH (c:Chunk {id: row.chunk_id})
MATCH (e:Entity {id: row.entity_id})
MERGE (c)-[:MENTIONS]->(e)
"""

CYPHER_QUESTIONS = """
UNWIND $rows AS row
MERGE (q:Question {id: row.id})
SET q.text = row.text,
    q.answer = row.answer,
    q.type = row.type,
    q.level = row.level,
    q.num_hops = row.num_hops,
    q.source = row.source,
    q.is_vi_exclusive = row.is_vi_exclusive,
    q.is_impossible = row.is_impossible,
    q.plausible_answer = row.plausible_answer,
    q.decomposition_json = row.decomposition_json
"""

CYPHER_SUPPORTED_BY = """
UNWIND $rows AS row
MATCH (q:Question {id: row.question_id})
MATCH (c:Chunk {id: row.chunk_id})
MERGE (q)-[:SUPPORTED_BY]->(c)
"""

CYPHER_BRIDGES = """
UNWIND $rows AS row
MATCH (q:Question {id: row.question_id})
MATCH (p:Page {id: row.page_id})
MERGE (q)-[:BRIDGES]->(p)
"""

_TYPED_LABELS = {"Person", "Organization", "Location", "Work"}


def _page_key(page_id: str | None, title: str) -> str:
    """Return a stable dedup key for a page (page_id when present, else title)."""
    if page_id:
        return f"pid:{page_id}"
    return f"title:{title.strip().lower()}"


def _page_node_id(page_id: str | None, title: str) -> str:
    """Return the stable Neo4j node id for a page."""
    if page_id:
        return str(page_id)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, title.strip().lower()))


def _chunk_node_id(page_key: str, segment_id: int) -> str:
    """Return a deterministic chunk id derived from page key + segment id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{page_key}#seg#{segment_id}"))


def _entity_node_id(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name.strip().lower()))


def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class _GraphBuilder:
    """Accumulates deduplicated Page/Chunk/Entity/Question rows across all input files."""

    def __init__(self, ner_backend: str) -> None:
        self.ner_backend = ner_backend
        self.pages: dict[str, dict] = {}
        self.chunks: dict[tuple[str, int], dict] = {}
        self.entities: dict[str, dict] = {}
        self.entity_labels: dict[str, list[str]] = {}
        self.mentions: set[tuple[str, str]] = set()
        self.questions: list[dict] = []
        self.supported_by: list[dict] = []
        self.bridges: list[dict] = []

        self.records_seen = 0
        self.records_skipped = 0

    def _ensure_page(self, page_id: str | None, title: str) -> str:
        key = _page_key(page_id, title)
        if key not in self.pages:
            node_id = _page_node_id(page_id, title)
            self.pages[key] = {
                "id": node_id,
                "title": title,
                "url": f"https://vi.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "summary": "",
            }
        return self.pages[key]["id"]

    def _ensure_chunk(self, page_key: str, page_node_id: str, segment: dict, title: str) -> str:
        segment_id = segment.get("segment_id")
        cache_key = (page_key, segment_id)
        if cache_key not in self.chunks:
            chunk_id = _chunk_node_id(page_key, segment_id)
            self.chunks[cache_key] = {
                "id": chunk_id,
                "page_id": page_node_id,
                "text": segment.get("text", ""),
                "seq": segment_id,
                "section": segment.get("paragraph_id"),
                "source_title": title,
                "source_segment_id": segment_id,
            }
            self._extract_entities_for_chunk(chunk_id, segment.get("text", ""))
        return self.chunks[cache_key]["id"]

    def _extract_entities_for_chunk(self, chunk_id: str, text: str) -> None:
        if not text.strip():
            return
        for name, entity_type in extract_entities(text):
            entity_key = name.strip().lower()
            if not entity_key:
                continue
            entity_id = _entity_node_id(name)
            if entity_key not in self.entities:
                self.entities[entity_key] = {"id": entity_id, "name": name.strip()}
                if entity_type in _TYPED_LABELS:
                    self.entity_labels.setdefault(entity_type, []).append(entity_id)
            self.mentions.add((chunk_id, entity_id))

    def add_record(self, record: dict) -> None:
        self.records_seen += 1

        context_pages: dict[str, str] = {}  # page_key -> node_id
        chunk_lookup: dict[tuple[str, int], str] = {}  # (title, segment_id) -> chunk_id

        for ctx in record.get("context", []):
            title = ctx.get("title", "").strip()
            if not title:
                continue
            page_id = ctx.get("page_id")
            page_key = _page_key(page_id, title)
            node_id = self._ensure_page(page_id, title)
            context_pages[page_key] = node_id

            for segment in ctx.get("segments", []):
                chunk_id = self._ensure_chunk(page_key, node_id, segment, title)
                chunk_lookup[(title, segment.get("segment_id"))] = chunk_id

        question_id = record.get("_id")
        if not question_id:
            self.records_skipped += 1
            return

        self.questions.append({
            "id": question_id,
            "text": record.get("question", ""),
            "answer": record.get("answer"),
            "type": record.get("type"),
            "level": record.get("level"),
            "num_hops": record.get("num_hops"),
            "source": record.get("source"),
            "is_vi_exclusive": bool(record.get("is_vi_exclusive", False)),
            "is_impossible": bool(record.get("is_impossible", False)),
            "plausible_answer": record.get("plausible_answer"),
            "decomposition_json": json.dumps(record.get("decomposition") or [], ensure_ascii=False),
        })

        for page_node_id in context_pages.values():
            self.bridges.append({"question_id": question_id, "page_id": page_node_id})

        for fact in record.get("supporting_facts", []):
            title = fact.get("title", "").strip()
            for segment_id in fact.get("segment_ids", []):
                chunk_id = chunk_lookup.get((title, segment_id))
                if chunk_id is None:
                    logger.warning(
                        "Unresolved supporting_fact segment",
                        extra={"question_id": question_id, "title": title, "segment_id": segment_id},
                    )
                    continue
                self.supported_by.append({"question_id": question_id, "chunk_id": chunk_id})

    def stats(self) -> dict:
        return {
            "records_seen": self.records_seen,
            "records_skipped": self.records_skipped,
            "pages": len(self.pages),
            "chunks": len(self.chunks),
            "entities": len(self.entities),
            "mentions": len(self.mentions),
            "questions": len(self.questions),
            "supported_by": len(self.supported_by),
            "bridges": len(self.bridges),
        }


def _load_in_batches(cypher: str, rows: list[dict], batch_size: int, label: str) -> None:
    if not rows:
        print(f"  {label}: 0 rows (skipped)")
        return
    total = neo4j_client.run_batch(cypher, rows, batch_size=batch_size)
    print(f"  {label}: {total:,} rows")


def ingest(
    input_paths: list[Path],
    limit: int | None,
    dry_run: bool,
    batch_size: int,
) -> dict:
    settings.ner_backend = "simple"  # zero-dependency backend, safe default for this task
    builder = _GraphBuilder(ner_backend=settings.ner_backend)

    total_processed = 0
    for path in input_paths:
        if not path.exists():
            logger.warning("Input file not found, skipping", extra={"path": str(path)})
            continue
        for record in _read_jsonl(path):
            if limit is not None and total_processed >= limit:
                break
            builder.add_record(record)
            total_processed += 1
            if total_processed % 500 == 0:
                logger.info("Ingest progress", extra={"processed": total_processed})
        if limit is not None and total_processed >= limit:
            break

    stats = builder.stats()
    print("\n=== Parsed dataset stats ===")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

    if dry_run:
        print("\nDry run: no writes to Neo4j.")
        return stats

    print("\n=== Ensuring schema ===")
    neo4j_client.setup_schema()

    print("\n=== Loading into Neo4j ===")
    _load_in_batches(CYPHER_PAGES, list(builder.pages.values()), batch_size, "Pages")
    _load_in_batches(CYPHER_CHUNKS, list(builder.chunks.values()), batch_size, "Chunks")
    _load_in_batches(CYPHER_ENTITIES, list(builder.entities.values()), batch_size, "Entities")

    for label, entity_ids in builder.entity_labels.items():
        cypher = CYPHER_ENTITY_LABEL % label
        rows = [{"entity_id": eid} for eid in entity_ids]
        _load_in_batches(cypher, rows, batch_size, f"Entity label :{label}")

    mention_rows = [{"chunk_id": c, "entity_id": e} for c, e in builder.mentions]
    _load_in_batches(CYPHER_MENTIONS, mention_rows, batch_size, "Mentions")

    _load_in_batches(CYPHER_QUESTIONS, builder.questions, batch_size, "Questions")
    _load_in_batches(CYPHER_SUPPORTED_BY, builder.supported_by, batch_size, "SUPPORTED_BY")
    _load_in_batches(CYPHER_BRIDGES, builder.bridges, batch_size, "BRIDGES")

    print("\nIngest complete.")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest multi-hop QA datasets into Neo4j")
    parser.add_argument("--seeds", default=str(_REPO_ROOT.parent / "seeds.jsonl"))
    parser.add_argument("--viquad", default=str(_REPO_ROOT.parent / "uit_viquad_seed42_8500.jsonl"))
    parser.add_argument("--viexclusive", default=str(_REPO_ROOT.parent / "vi_exclusive_batch1.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Max total records to process (across all files)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report stats without writing to Neo4j")
    parser.add_argument("--batch-size", type=int, default=1000, help="UNWIND batch size")
    args = parser.parse_args()

    input_paths = [Path(args.seeds), Path(args.viquad), Path(args.viexclusive)]
    ingest(input_paths, limit=args.limit, dry_run=args.dry_run, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
