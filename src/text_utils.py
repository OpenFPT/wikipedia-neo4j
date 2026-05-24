"""Text processing utilities for Vietnamese Wikipedia ingestion."""

from __future__ import annotations

import re
import unicodedata


def normalize_vietnamese(text: str) -> str:
    """Normalize Vietnamese text to NFC form and standardize whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
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
