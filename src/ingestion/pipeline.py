"""Ingestion pipelines for Wikipedia API and Hugging Face dataset."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Callable

import wikipedia
from datasets import load_dataset, load_from_disk

from src.infrastructure.llm import embed_texts
from src.logging_utils import get_logger
from src.infrastructure.neo4j_client import neo4j_client
from src.extraction.ner import extract_entities
from src.ingestion.text_utils import chunk_text_v2


logger = get_logger(__name__)


def _extract_aliases(text: str, entity_name: str) -> list[str]:
    """Extract aliases for an entity from parenthetical patterns in text."""
    aliases: list[str] = []
    pattern = re.compile(
        re.escape(entity_name) + r"\s*\(([^)]{2,60})\)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text[:2000]):
        candidate = match.group(1).strip()
        if candidate and candidate.lower() != entity_name.lower():
            aliases.append(candidate)
    return aliases[:5]


@dataclass
class IngestResult:
    """Summary result of a single ingested page/document."""

    topic: str
    page_id: str
    title: str
    url: str
    chunk_count: int
    entity_count: int


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Split text into overlapping chunks for retrieval and embedding."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    i = 0
    n = len(cleaned)
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append(cleaned[i:j])
        if j == n:
            break
        i = max(j - overlap, i + 1)
    return chunks


def _upsert_page_from_text(
    page_id: str,
    title: str,
    url: str,
    text: str,
    summary: str,
) -> IngestResult:
    """Upsert one page, chunks, entities, and mention edges into Neo4j."""
    chunks = chunk_text_v2(text, title=title)
    chunk_texts = [c.context + c.text for c in chunks]
    entities = extract_entities(text)
    embeddings = embed_texts(chunk_texts) if chunk_texts else []

    with neo4j_client.session() as session:
        session.run(
            """
            MERGE (p:Page {id: $id})
            SET p.title = $title,
                p.url = $url,
                p.summary = $summary
            """,
            id=page_id,
            title=title,
            url=url,
            summary=summary,
        )

        for idx, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{page_id}#chunk#{idx}"))
            embedding = embeddings[idx] if idx < len(embeddings) else []
            session.run(
                """
                MATCH (p:Page {id: $page_id})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.sequence_number = $seq,
                    c.section = $section,
                    c.embedding = $embedding
                MERGE (p)-[:HAS_CHUNK]->(c)
                """,
                page_id=page_id,
                chunk_id=chunk_id,
                text=chunk.text,
                seq=idx,
                section=chunk.section,
                embedding=embedding,
            )

        for name, entity_type in entities:
            entity_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name.lower()))
            label = entity_type if entity_type in ("Person", "Organization", "Location", "Work", "Event") else "Entity"
            aliases = _extract_aliases(text, name)
            session.run(
                f"""
                MERGE (e:{label} {{id: $entity_id}})
                SET e.name = $name,
                    e.aliases = $aliases
                """,
                entity_id=entity_id,
                name=name,
                aliases=aliases,
            )
            session.run(
                """
                MATCH (p:Page {id: $page_id})-[:HAS_CHUNK]->(c:Chunk)
                WHERE toLower(c.text) CONTAINS toLower($name)
                MATCH (e {id: $entity_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                page_id=page_id,
                entity_id=entity_id,
                name=name,
            )

    logger.info(
        "Page upserted",
        extra={"page_id": page_id, "title": title, "chunks": len(chunks), "entities": len(entities)},
    )

    return IngestResult(
        topic=title,
        page_id=page_id,
        title=title,
        url=url,
        chunk_count=len(chunks),
        entity_count=len(entities),
    )


def _write_page_links(page_id: str, linked_titles: list[str]) -> None:
    """Write LINKS_TO edges from a page to linked Wikipedia pages."""
    with neo4j_client.session() as session:
        for linked_title in linked_titles:
            linked_url = f"https://en.wikipedia.org/wiki/{linked_title.replace(' ', '_')}"
            linked_page_id = str(uuid.uuid5(uuid.NAMESPACE_URL, linked_url))
            session.run(
                """
                MATCH (p:Page {id: $source_id})
                MERGE (t:Page {id: $target_id})
                ON CREATE SET t.title = $target_title,
                              t.url = $target_url,
                              t.summary = coalesce(t.summary, '')
                MERGE (p)-[:LINKS_TO]->(t)
                """,
                source_id=page_id,
                target_id=linked_page_id,
                target_title=linked_title,
                target_url=linked_url,
            )


def ingest_topic(topic: str) -> IngestResult:
    """Ingest one topic from Wikipedia API into graph."""
    try:
        page = wikipedia.page(topic, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as exc:
        raise ValueError(f"Ambiguous topic '{topic}': {exc.options[:5]}") from exc
    except wikipedia.exceptions.PageError as exc:
        raise ValueError(f"Wikipedia page not found: '{topic}'") from exc

    page_id = str(uuid.uuid5(uuid.NAMESPACE_URL, page.url))

    try:
        summary = wikipedia.summary(topic, auto_suggest=False, sentences=3)
    except Exception:
        summary = page.content[:400]

    result = _upsert_page_from_text(
        page_id=page_id,
        title=page.title,
        url=page.url,
        text=page.content,
        summary=summary,
    )

    _write_page_links(page_id, page.links[:50])

    logger.info("Wikipedia topic ingested", extra={"topic": topic, "page_id": page_id})

    return IngestResult(
        topic=topic,
        page_id=page_id,
        title=page.title,
        url=page.url,
        chunk_count=result.chunk_count,
        entity_count=result.entity_count,
    )


def ingest_from_hf(
    config_name: str = "cleaned",
    split: str = "train",
    sample_size: int = 5,
    streaming: bool = True,
    on_progress: Callable[[int, int | None, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    local_path: str | None = None,
    dataset_id: str = "Keithsel/viwiki-20260523",
) -> list[IngestResult]:
    """Ingest records from a HuggingFace dataset (remote or local).

    If *local_path* is provided, loads from a pre-downloaded Arrow dataset
    directory (e.g. ``data/viet-wikipedia``). Otherwise loads from the
    specified HuggingFace dataset (default: Keithsel/viwiki-20260523).
    """
    results: list[IngestResult] = []
    total: int | None = sample_size if streaming else None

    if local_path:
        ds = load_from_disk(local_path)
        total = min(sample_size, len(ds))
        iterable = ds.select(range(total))
    elif streaming:
        iterable = load_dataset(dataset_id, config_name, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_id, config_name, split=split)
        total = min(sample_size, len(ds))
        iterable = ds.select(range(total))

    processed = 0

    for idx, raw_row in enumerate(iterable):
        if should_stop and should_stop():
            logger.info("HF ingestion stop requested", extra={"processed": processed})
            break
        if streaming and idx >= sample_size:
            break

        try:
            row = raw_row if isinstance(raw_row, dict) else dict(raw_row)
            page_id = str(row.get("id", ""))
            title = str(row.get("title", "")).strip() or f"untitled-{uuid.uuid4()}"
            url = str(row.get("url", "")).strip() or f"https://example.org/{title.replace(' ', '_')}"
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            summary = text[:400]
            if not page_id:
                page_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url))
            result = _upsert_page_from_text(
                page_id=page_id,
                title=title,
                url=url,
                text=text,
                summary=summary,
            )
        except Exception as exc:
            logger.warning("Skipping malformed HF row", extra={"index": idx, "error": str(exc)})
            continue

        results.append(result)
        processed += 1
        if on_progress:
            on_progress(processed, total, result.title)

    logger.info("HF ingestion completed", extra={"processed": processed, "requested": sample_size})
    return results
