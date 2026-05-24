from __future__ import annotations

import src.ingest as ingest
import src.ner as ner


def test_chunk_text_empty() -> None:
    assert ingest._chunk_text("   ") == []


def test_chunk_text_overlap_behavior() -> None:
    text = "a" * 200
    chunks = ingest._chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert chunks[0] == "a" * 50


def test_extract_entities_simple() -> None:
    text = "Neo4j is used with New York teams at Google Cloud."
    entities = ner._extract_entities_simple(text)
    assert "New York" in entities or "Google Cloud" in entities
