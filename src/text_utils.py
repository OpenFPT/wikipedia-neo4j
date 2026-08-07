"""Backward-compatibility shim: imports redirect to src.ingestion.text_utils."""

from src.ingestion.text_utils import (  # noqa: F401
    normalize_vietnamese,
    chunk_text,
    chunk_text_v2,
    ChunkV2,
    strip_wiki_markup,
    extract_wikilinks,
    entity_grounded_in_text,
)
