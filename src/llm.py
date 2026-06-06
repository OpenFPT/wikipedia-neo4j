"""Backward-compatibility shim: imports redirect to src.infrastructure.llm."""

from src.infrastructure.llm import (  # noqa: F401
    embed_texts,
    embed_texts_batch,
    generate_readonly_cypher,
    assert_readonly_cypher,
)
