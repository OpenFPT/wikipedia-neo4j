"""Text processing utilities for Vietnamese Wikipedia ingestion."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


def normalize_vietnamese(text: str) -> str:
    """Normalize Vietnamese text to NFC form and standardize whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Split text into overlapping chunks for retrieval and embedding.

    Legacy interface kept for backward compatibility.
    """
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


# ---------------------------------------------------------------------------
# Structure-aware chunking (v2)
# ---------------------------------------------------------------------------

_VIET_ABBREVS = re.compile(
    r"\b(?:TP|PGS|TS|GS|ThS|CN|Bs|KTS|TSKH|Th\.S|vs|St)\.$"
)

_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkV2:
    text: str
    section: str = ""
    page_title: str = ""
    seq: int = 0
    context: str = field(default="", repr=False)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting Vietnamese abbreviations."""
    parts = _SENT_SPLIT.split(text)
    sentences: list[str] = []
    buf = ""
    for part in parts:
        if buf:
            buf = buf + " " + part
        else:
            buf = part
        if _VIET_ABBREVS.search(buf):
            continue
        sentences.append(buf)
        buf = ""
    if buf:
        sentences.append(buf)
    return sentences


def _parse_sections(text: str) -> list[tuple[str, str]]:
    """Parse Wikipedia wikitext into (heading_path, body) pairs."""
    sections: list[tuple[str, str]] = []
    heading_stack: list[tuple[int, str]] = []

    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        return [("", text)]

    intro = text[: matches[0].start()].strip()
    if intro:
        sections.append(("", intro))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))

        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        path = " > ".join(h[1] for h in heading_stack)
        if body:
            sections.append((path, body))

    return sections


def _recursive_split(
    text: str,
    max_size: int,
    min_size: int,
) -> list[str]:
    """Recursively split text: paragraphs -> sentences -> words -> chars."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) > 1:
        return _merge_splits(paragraphs, max_size, min_size)

    sentences = _split_sentences(text)
    if len(sentences) > 1:
        return _merge_splits(sentences, max_size, min_size, joiner=" ")

    words = text.split()
    if len(words) > 1:
        return _merge_splits(words, max_size, min_size, joiner=" ")

    chunks = []
    for i in range(0, len(text), max_size):
        chunks.append(text[i : i + max_size])
    return chunks


def _merge_splits(
    parts: list[str],
    max_size: int,
    min_size: int,
    joiner: str = "\n\n",
) -> list[str]:
    """Merge small parts together, split large parts recursively."""
    chunks: list[str] = []
    buf = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        candidate = (buf + joiner + part) if buf else part

        if len(candidate) <= max_size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            if len(part) > max_size:
                sub_chunks = _recursive_split(part, max_size, min_size)
                chunks.extend(sub_chunks)
                buf = ""
            else:
                buf = part

    if buf:
        chunks.append(buf)

    if len(chunks) >= 2 and len(chunks[-1]) < min_size:
        merged = chunks[-2] + joiner + chunks[-1]
        if len(merged) <= max_size:
            chunks[-2] = merged
            chunks.pop()

    return chunks


def chunk_text_v2(
    text: str,
    title: str = "",
    max_chunk_size: int = 2048,
    target_chunk_size: int = 1600,
    min_chunk_size: int = 200,
    include_context: bool = True,
) -> list[ChunkV2]:
    """Structure-aware recursive chunking for Vietnamese Wikipedia.

    Splits on section headings, then paragraphs, then sentences.
    Returns rich chunk objects with section metadata.
    """
    if not text or not text.strip():
        return []

    sections = _parse_sections(text)
    chunks: list[ChunkV2] = []
    seq = 0

    for section_path, body in sections:
        body = re.sub(r"[ \t]+", " ", body).strip()
        if not body:
            continue

        split_texts = _recursive_split(body, target_chunk_size, min_chunk_size)

        for chunk_text_str in split_texts:
            chunk_text_str = chunk_text_str.strip()
            if not chunk_text_str:
                continue

            context = ""
            if include_context and (title or section_path):
                ctx_parts = []
                if title:
                    ctx_parts.append(f"Page: {title}")
                if section_path:
                    ctx_parts.append(f"Section: {section_path}")
                context = "[" + " | ".join(ctx_parts) + "] "

            chunks.append(ChunkV2(
                text=chunk_text_str,
                section=section_path,
                page_title=title,
                seq=seq,
                context=context,
            ))
            seq += 1

    return chunks


# ---------------------------------------------------------------------------
# Wiki link extraction
# ---------------------------------------------------------------------------

_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")

_SKIP_PREFIXES = (
    "File:", "Image:", "Category:", "Template:", "Wikipedia:",
    "Tập tin:", "Hình:", "Thể loại:", "Bản mẫu:",
    "wikt:", "s:", "q:", "b:", "n:", "v:", "commons:",
)


def extract_wikilinks(raw_text: str) -> list[tuple[str, str]]:
    """Extract wiki links from raw Wikipedia markup.

    Returns deduplicated list of (target_title, display_text) tuples.
    Skips File/Category/Template links and strips anchor fragments.
    """
    seen: set[str] = set()
    results: list[tuple[str, str]] = []

    for match in _WIKILINK_RE.finditer(raw_text):
        content = match.group(1).strip()
        if not content:
            continue

        if "|" in content:
            target, display = content.split("|", 1)
        else:
            target = content
            display = content

        target = target.strip()
        display = display.strip()

        if not target or not display:
            continue

        if any(target.startswith(p) or target.lower().startswith(p.lower()) for p in _SKIP_PREFIXES):
            continue

        if "#" in target:
            target = target.split("#", 1)[0].strip()
        if not target:
            continue

        target = target[0].upper() + target[1:] if target else target
        target = target.replace("_", " ")

        key = target.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append((target, display))

    return results


def entity_grounded_in_text(
    entity_name: str,
    display_text: str,
    chunk_text: str,
) -> bool:
    """Check if an entity (or its display form) actually appears in chunk text.

    Handles exact match, display text match, partial name match for
    multi-word entities, and word-boundary checks for short names.
    """
    import unicodedata

    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKC", s).lower().strip()

    chunk_lower = _norm(chunk_text)
    name_lower = _norm(entity_name)
    display_lower = _norm(display_text) if display_text else name_lower

    if not name_lower:
        return False

    # Exact match of full entity name
    if name_lower in chunk_lower:
        return True

    # Display text match (wikilink [[target|display]] form)
    if display_lower != name_lower and display_lower in chunk_lower:
        return True

    # Multi-word: check if last 2 tokens appear together
    tokens = name_lower.split()
    if len(tokens) >= 2:
        last_two = " ".join(tokens[-2:])
        if last_two in chunk_lower:
            return True
        # Also check first+last for Vietnamese names (Họ + Tên)
        first_last = f"{tokens[0]} {tokens[-1]}"
        if len(tokens) >= 3 and first_last in chunk_lower:
            return True

    # Single-token: require word boundary match to avoid "Năm" in "năm 2017"
    if len(tokens) == 1 and len(name_lower) >= 3:
        import re
        pattern = r"(?<!\w)" + re.escape(name_lower) + r"(?!\w)"
        if re.search(pattern, chunk_lower):
            return True

    return False
