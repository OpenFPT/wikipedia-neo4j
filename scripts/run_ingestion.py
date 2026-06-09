"""Full ingestion pipeline: paragraphs → NER → entity resolution → Neo4j."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pyarrow.parquet as pq
from neo4j import GraphDatabase

from src.config import settings
from src.extraction.entity_resolution import EntityResolver
from src.logging_utils import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PARQUET = DATA_DIR / "viwiki_paragraphs.parquet"


@dataclass
class NERMention:
    """A named entity mention extracted from text."""

    text: str
    entity_type: str  # Person, Organization, Location, Work, Event
    start: int = 0
    end: int = 0


class NERPipeline(Protocol):
    """Interface for NER extraction — swap in AnhNQ's implementation later."""

    def extract(self, text: str) -> list[NERMention]: ...


def _classify_entity_type(name: str) -> str:
    """Rule-based entity type classification for Vietnamese + English."""
    lowered = name.lower()
    org_keywords = [
        "company", "co.", "corporation", "corp", "inc", "ltd",
        "university", "college", "bank", "ministry", "department",
        "tập đoàn", "công ty", "ngân hàng", "đại học", "trường ",
        "học viện", "bộ ", "sở ", "ủy ban",
    ]
    location_keywords = [
        "city", "province", "district", "county", "state",
        "river", "mount", "mountain", "lake", "sea", "island", "bay",
        "thành phố", "tỉnh", "quận", "huyện", "xã",
        "sông", "núi", "hồ", "biển", "đảo", "vịnh",
    ]
    work_keywords = [
        "film", "movie", "novel", "book", "album", "song",
        "phim", "tiểu thuyết", "tác phẩm", "bài hát",
    ]

    if any(kw in lowered for kw in org_keywords):
        return "Organization"
    if any(kw in lowered for kw in location_keywords):
        return "Location"
    if any(kw in lowered for kw in work_keywords):
        return "Work"
    word_count = len(name.split())
    if 1 < word_count <= 4:
        return "Person"
    return "Unknown"


class BasicVietnameseNER:
    """Regex-based NER fallback for Vietnamese text."""

    _UPPER_PATTERN = re.compile(
        r"\b([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){1,5})\b"
    )

    def extract(self, text: str) -> list[NERMention]:
        mentions = []
        seen: set[str] = set()
        for match in self._UPPER_PATTERN.finditer(text):
            name = match.group(1).strip()
            key = name.lower()
            if key in seen or len(name) < 3:
                continue
            seen.add(key)
            mentions.append(NERMention(
                text=name,
                entity_type=_classify_entity_type(name),
                start=match.start(),
                end=match.end(),
            ))
        return mentions


class UndertheseaNER:
    """NER using underthesea library (BIO tagging)."""

    def extract(self, text: str) -> list[NERMention]:
        try:
            from underthesea import ner as under_ner
        except ImportError:
            logger.warning("underthesea not installed, falling back to BasicVietnameseNER")
            return BasicVietnameseNER().extract(text)

        tags = under_ner(text)
        mentions: list[NERMention] = []
        buffer: list[str] = []
        buffer_type: str = "Unknown"

        type_map = {"PER": "Person", "ORG": "Organization", "LOC": "Location", "MISC": "Work"}

        for token, _pos, _chunk, ner_tag in tags:
            if ner_tag == "O":
                if buffer:
                    name = " ".join(buffer)
                    mentions.append(NERMention(text=name, entity_type=buffer_type))
                    buffer = []
                    buffer_type = "Unknown"
                continue
            tag, raw_type = ner_tag.split("-", 1)
            mapped_type = type_map.get(raw_type, "Unknown")
            if tag == "B":
                if buffer:
                    name = " ".join(buffer)
                    mentions.append(NERMention(text=name, entity_type=buffer_type))
                buffer = [token]
                buffer_type = mapped_type
            elif tag == "I" and buffer:
                buffer.append(token)
            else:
                if buffer:
                    name = " ".join(buffer)
                    mentions.append(NERMention(text=name, entity_type=buffer_type))
                buffer = [token]
                buffer_type = mapped_type

        if buffer:
            name = " ".join(buffer)
            mentions.append(NERMention(text=name, entity_type=buffer_type))

        # Deduplicate
        seen: set[str] = set()
        deduped: list[NERMention] = []
        for m in mentions:
            key = m.text.strip().lower()
            if key in seen or not key:
                continue
            seen.add(key)
            deduped.append(m)
        return deduped


def get_ner_pipeline() -> NERPipeline:
    """Factory: return NER pipeline based on config."""
    backend = getattr(settings, "ner_backend", "simple")
    if backend == "underthesea":
        return UndertheseaNER()
    return BasicVietnameseNER()


@dataclass
class IngestStats:
    """Statistics from an ingestion run."""

    articles_processed: int = 0
    paragraphs_written: int = 0
    entities_created: int = 0
    relationships_created: int = 0
    errors: int = 0


def _entity_id(name: str) -> str:
    return hashlib.md5(name.strip().lower().encode()).hexdigest()


def _write_article_batch(
    tx,
    articles: list[dict],
) -> None:
    """Write a batch of articles to Neo4j."""
    tx.run(
        """
        UNWIND $articles AS art
        MERGE (a:Article {id: art.id})
        SET a.title = art.title, a.url = art.url
        """,
        articles=articles,
    )


def _write_paragraph_batch(
    tx,
    paragraphs: list[dict],
) -> None:
    """Write a batch of paragraphs and link to articles."""
    tx.run(
        """
        UNWIND $paragraphs AS para
        MERGE (p:Paragraph {id: para.id})
        SET p.text = para.text,
            p.article_id = para.article_id,
            p.paragraph_index = para.paragraph_index
        WITH p, para
        MATCH (a:Article {id: para.article_id})
        MERGE (a)-[:HAS_PARAGRAPH]->(p)
        """,
        paragraphs=paragraphs,
    )


def _write_entity_batch(
    tx,
    entities: list[dict],
) -> None:
    """Write entities with typed labels."""
    for entity in entities:
        label = entity["label"]
        tx.run(
            f"""
            MERGE (e:{label} {{id: $id}})
            SET e.name = $name,
                e.aliases_text = $aliases_text
            """,
            id=entity["id"],
            name=entity["name"],
            aliases_text=entity["aliases_text"],
        )


def _write_mention_edges(
    tx,
    edges: list[dict],
) -> None:
    """Create MENTIONS edges between paragraphs and entities."""
    for edge in edges:
        label = edge["entity_label"]
        tx.run(
            f"""
            MATCH (p:Paragraph {{id: $para_id}})
            MATCH (e:{label} {{id: $entity_id}})
            MERGE (p)-[:MENTIONS]->(e)
            """,
            para_id=edge["para_id"],
            entity_id=edge["entity_id"],
        )


VALID_LABELS = {"Person", "Organization", "Location", "Work", "Event", "Unknown"}


def _safe_label(entity_type: str) -> str:
    if entity_type in VALID_LABELS:
        return entity_type
    return "Unknown"


def run_ingestion(
    parquet_path: Path = DEFAULT_PARQUET,
    ner: NERPipeline | None = None,
    batch_size: int = 500,
    max_articles: int | None = None,
) -> IngestStats:
    """Run the full ingestion pipeline from parquet to Neo4j."""
    if ner is None:
        ner = get_ner_pipeline()

    resolver = EntityResolver()
    stats = IngestStats()

    logger.info("Reading parquet", extra={"path": str(parquet_path)})
    table = pq.read_table(parquet_path)
    df_rows = table.to_pydict()
    total_rows = len(df_rows["paragraph_id"])

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )

    article_batch: list[dict] = []
    para_batch: list[dict] = []
    entity_batch: list[dict] = []
    edge_batch: list[dict] = []
    seen_articles: set[str] = set()

    for i in range(total_rows):
        if max_articles and stats.articles_processed >= max_articles:
            break

        try:
            article_id = df_rows["article_id"][i]
            para_id = df_rows["paragraph_id"][i]
            title = df_rows["title"][i]
            url = df_rows["url"][i]
            para_idx = df_rows["paragraph_index"][i]
            text = df_rows["text"][i]

            if article_id not in seen_articles:
                seen_articles.add(article_id)
                article_batch.append({"id": article_id, "title": title, "url": url})
                stats.articles_processed += 1

            para_batch.append({
                "id": para_id,
                "text": text,
                "article_id": article_id,
                "paragraph_index": para_idx,
            })
            stats.paragraphs_written += 1

            mentions = ner.extract(text)
            for mention in mentions:
                resolved = resolver.resolve(mention.text, mention.entity_type)
                label = _safe_label(resolved.entity_type)
                eid = _entity_id(resolved.canonical_name)

                entity_batch.append({
                    "id": eid,
                    "name": resolved.canonical_name,
                    "label": label,
                    "aliases_text": ", ".join(resolved.aliases) if resolved.aliases else "",
                })
                edge_batch.append({
                    "para_id": para_id,
                    "entity_id": eid,
                    "entity_label": label,
                })
                stats.entities_created += 1
                stats.relationships_created += 1

            if len(para_batch) >= batch_size:
                _flush_batch(driver, article_batch, para_batch, entity_batch, edge_batch)
                article_batch.clear()
                para_batch.clear()
                entity_batch.clear()
                edge_batch.clear()

                if stats.paragraphs_written % 5000 == 0:
                    logger.info("Progress", extra={
                        "paragraphs": stats.paragraphs_written,
                        "articles": stats.articles_processed,
                        "total_rows": total_rows,
                    })

        except Exception as e:
            stats.errors += 1
            if stats.errors <= 10:
                logger.warning("Row error", extra={"index": i, "error": str(e)})

    # Flush remaining
    if para_batch:
        _flush_batch(driver, article_batch, para_batch, entity_batch, edge_batch)

    driver.close()
    logger.info("Ingestion complete", extra={
        "articles": stats.articles_processed,
        "paragraphs": stats.paragraphs_written,
        "entities": stats.entities_created,
        "errors": stats.errors,
    })
    return stats


def _flush_batch(
    driver,
    articles: list[dict],
    paragraphs: list[dict],
    entities: list[dict],
    edges: list[dict],
) -> None:
    """Write all pending batches to Neo4j in a single transaction."""
    with driver.session() as session:
        if articles:
            session.execute_write(_write_article_batch, articles)
        if paragraphs:
            session.execute_write(_write_paragraph_batch, paragraphs)
        if entities:
            session.execute_write(_write_entity_batch, entities)
        if edges:
            session.execute_write(_write_mention_edges, edges)


def main() -> None:
    """CLI entry point for ingestion."""
    import argparse

    parser = argparse.ArgumentParser(description="Run ViWiki ingestion pipeline")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-articles", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    stats = run_ingestion(
        parquet_path=args.parquet,
        batch_size=args.batch_size,
        max_articles=args.max_articles,
    )
    print(f"Done: {stats.articles_processed} articles, {stats.paragraphs_written} paragraphs, "
          f"{stats.entities_created} entities, {stats.errors} errors")


if __name__ == "__main__":
    main()
