"""Backward-compatibility shim: imports redirect to src.extraction.relations."""

from src.extraction.relations import (  # noqa: F401
    RELATION_TYPES,
    Triple,
    extract_relations,
    _parse_triples,
    extract_relations_batch,
)
