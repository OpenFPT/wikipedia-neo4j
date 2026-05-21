"""Ingestion pipelines for Wikipedia API and Hugging Face dataset."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Callable

import wikipedia
from datasets import load_dataset

from src.llm import embed_texts
from src.logging_utils import get_logger
from src.neo4j_client import neo4j_client
from src.config import settings


logger = get_logger(__name__)


@dataclass
class IngestResult:
    """Summary result of a single ingested page/document."""

    topic: str
    page_id: str
    title: str
    url: str
    chunk_count: int
    entity_count: int


def _upsert_page_from_text(
    page_id: str,
    title: str,
    url: str,
    text: str,
    summary: str,
) -> IngestResult:
    """Upsert one page, chunks, entities, and mention edges into Neo4j."""
    chunks = _chunk_text(text)
    entities = _extract_entities(text)
    embeddings = embed_texts(chunks) if chunks else []

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
                    c.embedding = $embedding
                MERGE (p)-[:HAS_CHUNK]->(c)
                """,
                page_id=page_id,
                chunk_id=chunk_id,
                text=chunk,
                seq=idx,
                embedding=embedding,
            )

        for name, entity_type in entities:
            entity_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name.lower()))
            session.run(
                """
                MERGE (e:Entity {id: $entity_id})
                SET e.name = $name,
                    e.type = $entity_type
                FOREACH (_ IN CASE WHEN $entity_type = 'Person' THEN [1] ELSE [] END | SET e:Person)
                FOREACH (_ IN CASE WHEN $entity_type = 'Organization' THEN [1] ELSE [] END | SET e:Organization)
                FOREACH (_ IN CASE WHEN $entity_type = 'Location' THEN [1] ELSE [] END | SET e:Location)
                FOREACH (_ IN CASE WHEN $entity_type = 'Work' THEN [1] ELSE [] END | SET e:Work)
                """,
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
            )
            session.run(
                """
                MATCH (p:Page {id: $page_id})-[:HAS_CHUNK]->(c:Chunk)
                WHERE toLower(c.text) CONTAINS toLower($name)
                MATCH (e:Entity {id: $entity_id})
                MERGE (c)-[:MENTIONS]->(e)
                FOREACH (_ IN CASE WHEN $entity_type = 'Person' THEN [1] ELSE [] END | MERGE (c)-[:MENTIONS_PERSON]->(e))
                FOREACH (_ IN CASE WHEN $entity_type = 'Organization' THEN [1] ELSE [] END | MERGE (c)-[:MENTIONS_ORG]->(e))
                FOREACH (_ IN CASE WHEN $entity_type = 'Location' THEN [1] ELSE [] END | MERGE (c)-[:MENTIONS_LOCATION]->(e))
                FOREACH (_ IN CASE WHEN $entity_type = 'Work' THEN [1] ELSE [] END | MERGE (c)-[:MENTIONS_WORK]->(e))
                """,
                page_id=page_id,
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
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


def _extract_entities_simple(text: str, max_entities: int = 25) -> list[str]:
    """Extract simple title-cased entities using regex heuristic."""
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    deduped: list[str] = []
    seen = set()
    for c in candidates:
        key = c.strip().lower()
        if len(c) < 3 or key in seen:
            continue
        seen.add(key)
        deduped.append(c.strip())
        if len(deduped) >= max_entities:
            break
    return deduped


def _extract_entities_underthesea(text: str, max_entities: int = 25) -> list[str]:
    try:
        from underthesea import ner as under_ner
    except Exception:
        return _extract_entities_simple(text, max_entities)

    tags = under_ner(text)
    entities: list[str] = []
    buffer: list[str] = []
    for token, _pos, _chunk, ner_tag in tags:
        if ner_tag == "O":
            if buffer:
                entities.append(" ".join(buffer))
                buffer = []
            continue
        tag = ner_tag.split("-", 1)[0]
        if tag == "B":
            if buffer:
                entities.append(" ".join(buffer))
            buffer = [token]
        elif tag == "I" and buffer:
            buffer.append(token)
        else:
            if buffer:
                entities.append(" ".join(buffer))
            buffer = [token]

    if buffer:
        entities.append(" ".join(buffer))

    deduped: list[str] = []
    seen = set()
    for ent in entities:
        key = ent.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(ent.strip())
        if len(deduped) >= max_entities:
            break
    return deduped


_phonlp_model = None
_phonlp_segmenter = None


def _get_phonlp():
    global _phonlp_model
    if _phonlp_model is None:
        import phonlp

        phonlp.download(save_dir=settings.phonlp_model_dir)
        _phonlp_model = phonlp.load(save_dir=settings.phonlp_model_dir)
    return _phonlp_model


def _get_vncorenlp_segmenter():
    global _phonlp_segmenter
    if _phonlp_segmenter is None:
        import py_vncorenlp

        py_vncorenlp.download_model(save_dir=settings.vncorenlp_dir)
        _phonlp_segmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=settings.vncorenlp_dir,
        )
    return _phonlp_segmenter


def _extract_entities_phonlp(text: str, max_entities: int = 25) -> list[tuple[str, str]]:
    try:
        model = _get_phonlp()
        segmenter = _get_vncorenlp_segmenter()
    except Exception:
        return [(e, _classify_entity_type(e)) for e in _extract_entities_simple(text, max_entities)]

    segmented = segmenter.word_segment(text)
    if not segmented:
        return [(e, _classify_entity_type(e)) for e in _extract_entities_simple(text, max_entities)]
    result = model.annotate(text=segmented[0])
    entities: list[tuple[str, str]] = []
    buffer: list[str] = []
    buffer_type: str | None = None
    type_map = {
        "PER": "Person",
        "ORG": "Organization",
        "LOC": "Location",
        "MISC": "Work",
    }
    for word, _pos, _chunk, ner_tag in result.get("ner", []):
        if ner_tag == "O":
            if buffer:
                entities.append((" ".join(buffer), buffer_type or "Unknown"))
                buffer = []
                buffer_type = None
            continue
        tag, raw_type = ner_tag.split("-", 1)
        mapped_type = type_map.get(raw_type, "Unknown")
        if tag == "B":
            if buffer:
                entities.append((" ".join(buffer), buffer_type or "Unknown"))
            buffer = [word]
            buffer_type = mapped_type
        elif tag == "I" and buffer:
            buffer.append(word)
        else:
            if buffer:
                entities.append((" ".join(buffer), buffer_type or "Unknown"))
            buffer = [word]
            buffer_type = mapped_type

    if buffer:
        entities.append((" ".join(buffer), buffer_type or "Unknown"))

    deduped: list[tuple[str, str]] = []
    seen = set()
    for ent, ent_type in entities:
        key = ent.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((ent.strip(), ent_type))
        if len(deduped) >= max_entities:
            break
    return deduped


def _extract_entities(text: str, max_entities: int = 25) -> list[tuple[str, str]]:
    """Extract entities and return (name, type) tuples regardless of backend."""
    if settings.ner_backend == "underthesea":
        names = _extract_entities_underthesea(text, max_entities)
        return [(n, _classify_entity_type(n)) for n in names]
    if settings.ner_backend == "phonlp":
        return _extract_entities_phonlp(text, max_entities)
    names = _extract_entities_simple(text, max_entities)
    return [(n, _classify_entity_type(n)) for n in names]


def _classify_entity_type(name: str) -> str:
    lowered = name.lower()
    org_keywords = [
        "company",
        "co.",
        "corporation",
        "corp",
        "inc",
        "ltd",
        "university",
        "college",
        "bank",
        "ministry",
        "department",
        "tập đoàn",
        "công ty",
        "ngân hàng",
        "đại học",
        "trường ",
        "học viện",
        "bộ ",
        "sở ",
        "ủy ban",
    ]
    location_keywords = [
        "city",
        "province",
        "district",
        "county",
        "state",
        "river",
        "mount",
        "mountain",
        "lake",
        "sea",
        "island",
        "bay",
        "thành phố",
        "tỉnh",
        "quận",
        "huyện",
        "xã",
        "sông",
        "núi",
        "hồ",
        "biển",
        "đảo",
        "vịnh",
    ]
    work_keywords = [
        "film",
        "movie",
        "novel",
        "book",
        "album",
        "song",
        "phim",
        "tiểu thuyết",
        "tác phẩm",
        "bài hát",
    ]

    if any(keyword in lowered for keyword in org_keywords):
        return "Organization"
    if any(keyword in lowered for keyword in location_keywords):
        return "Location"
    if any(keyword in lowered for keyword in work_keywords):
        return "Work"
    word_count = len(name.split())
    if 1 < word_count <= 4:
        return "Person"
    return "Unknown"


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
    chunks = _chunk_text(page.content)
    entities = _extract_entities(page.content)
    embeddings = embed_texts(chunks) if chunks else []
    linked_titles = page.links[:50]

    with neo4j_client.session() as session:
        session.run(
            """
            MERGE (p:Page {id: $id})
            SET p.title = $title,
                p.url = $url,
                p.summary = $summary
            """,
            id=page_id,
            title=page.title,
            url=page.url,
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
                    c.embedding = $embedding
                MERGE (p)-[:HAS_CHUNK]->(c)
                """,
                page_id=page_id,
                chunk_id=chunk_id,
                text=chunk,
                seq=idx,
                embedding=embedding,
            )

        for name, entity_type in entities:
            entity_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name.lower()))
            session.run(
                """
                MERGE (e:Entity {id: $entity_id})
                SET e.name = $name,
                    e.type = $entity_type
                FOREACH (_ IN CASE WHEN $entity_type = 'Person' THEN [1] ELSE [] END | SET e:Person)
                FOREACH (_ IN CASE WHEN $entity_type = 'Organization' THEN [1] ELSE [] END | SET e:Organization)
                FOREACH (_ IN CASE WHEN $entity_type = 'Location' THEN [1] ELSE [] END | SET e:Location)
                FOREACH (_ IN CASE WHEN $entity_type = 'Work' THEN [1] ELSE [] END | SET e:Work)
                """,
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
            )
            session.run(
                """
                MATCH (p:Page {id: $page_id})-[:HAS_CHUNK]->(c:Chunk)
                WHERE toLower(c.text) CONTAINS toLower($name)
                MATCH (e:Entity {id: $entity_id})
                MERGE (c)-[:MENTIONS]->(e)
                FOREACH (_ IN CASE WHEN $entity_type = 'Person' THEN [1] ELSE [] END | MERGE (c)-[:MENTIONS_PERSON]->(e))
                FOREACH (_ IN CASE WHEN $entity_type = 'Organization' THEN [1] ELSE [] END | MERGE (c)-[:MENTIONS_ORG]->(e))
                FOREACH (_ IN CASE WHEN $entity_type = 'Location' THEN [1] ELSE [] END | MERGE (c)-[:MENTIONS_LOCATION]->(e))
                FOREACH (_ IN CASE WHEN $entity_type = 'Work' THEN [1] ELSE [] END | MERGE (c)-[:MENTIONS_WORK]->(e))
                """,
                page_id=page_id,
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
            )

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

    logger.info("Wikipedia topic ingested", extra={"topic": topic, "page_id": page_id})

    return IngestResult(
        topic=topic,
        page_id=page_id,
        title=page.title,
        url=page.url,
        chunk_count=len(chunks),
        entity_count=len(entities),
    )


def ingest_from_hf(
    config_name: str = "20231101.en",
    split: str = "train",
    sample_size: int = 5,
    streaming: bool = True,
    on_progress: Callable[[int, int | None, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[IngestResult]:
    """Ingest sample records from `wikimedia/wikipedia` Hugging Face dataset."""
    results: list[IngestResult] = []
    total: int | None = sample_size if streaming else None

    if streaming:
        iterable = load_dataset("wikimedia/wikipedia", config_name, split=split, streaming=True)
    else:
        ds = load_dataset("wikimedia/wikipedia", config_name, split=split)
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
