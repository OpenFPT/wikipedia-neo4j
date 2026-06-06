"""Backward-compatibility shim: imports redirect to src.infrastructure.local_llm."""

from src.infrastructure.local_llm import (  # noqa: F401
    generate,
    chat,
)
