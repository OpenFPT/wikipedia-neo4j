"""Backward-compatibility shim: imports redirect to src.extraction.ner."""

from src.extraction.ner import (  # noqa: F401
    extract_entities,
    extract_entities_batch,
    classify_entity_type,
    normalize_entity,
    strip_disambiguation,
    postprocess_entities,
)
