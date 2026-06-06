"""Backward-compatibility shim: imports redirect to src.extraction.entity_resolution."""

from src.extraction.entity_resolution import (  # noqa: F401
    ResolvedEntity,
    KNOWN_ALIASES,
    remove_diacritics,
    normalize_key,
    normalize_key_no_diacritics,
    EntityResolver,
)
