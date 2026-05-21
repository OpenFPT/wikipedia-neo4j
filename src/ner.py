"""Named Entity Recognition backends and entity classification."""

from __future__ import annotations

import re

from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

_phonlp_model = None
_phonlp_segmenter = None

_NER_TYPE_MAP = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "MISC": "Work",
}

_ORG_KEYWORDS = [
    "company", "co.", "corporation", "corp", "inc", "ltd",
    "university", "college", "bank", "ministry", "department",
    "tập đoàn", "công ty", "ngân hàng", "đại học",
    "trường ", "học viện", "bộ ", "sở ", "ủy ban",
]

_LOCATION_KEYWORDS = [
    "city", "province", "district", "county", "state",
    "river", "mount", "mountain", "lake", "sea", "island", "bay",
    "thành phố", "tỉnh", "quận", "huyện", "xã",
    "sông", "núi", "hồ", "biển", "đảo", "vịnh",
]

_WORK_KEYWORDS = [
    "film", "movie", "novel", "book", "album", "song",
    "phim", "tiểu thuyết", "tác phẩm", "bài hát",
]


def classify_entity_type(name: str) -> str:
    """Classify an entity name into Person/Organization/Location/Work/Unknown."""
    lowered = name.lower()
    if any(kw in lowered for kw in _ORG_KEYWORDS):
        return "Organization"
    if any(kw in lowered for kw in _LOCATION_KEYWORDS):
        return "Location"
    if any(kw in lowered for kw in _WORK_KEYWORDS):
        return "Work"
    word_count = len(name.split())
    if 1 < word_count <= 4:
        return "Person"
    return "Unknown"


def _collect_bio_entities(
    tagged_tokens: list[tuple[str, str]],
    max_entities: int = 25,
) -> list[tuple[str, str]]:
    """Accumulate BIO-tagged (token, ner_tag) into deduplicated (name, type) pairs."""
    entities: list[tuple[str, str]] = []
    buffer: list[str] = []
    buffer_type: str | None = None

    for token, ner_tag in tagged_tokens:
        if ner_tag == "O":
            if buffer:
                entities.append((" ".join(buffer), buffer_type or "Unknown"))
                buffer = []
                buffer_type = None
            continue

        parts = ner_tag.split("-", 1)
        tag = parts[0]
        raw_type = parts[1] if len(parts) > 1 else None
        mapped_type = _NER_TYPE_MAP.get(raw_type, "Unknown") if raw_type else "Unknown"

        if tag == "B":
            if buffer:
                entities.append((" ".join(buffer), buffer_type or "Unknown"))
            buffer = [token]
            buffer_type = mapped_type
        elif tag == "I" and buffer:
            buffer.append(token)
        else:
            if buffer:
                entities.append((" ".join(buffer), buffer_type or "Unknown"))
            buffer = [token]
            buffer_type = mapped_type

    if buffer:
        entities.append((" ".join(buffer), buffer_type or "Unknown"))

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for ent, ent_type in entities:
        key = ent.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((ent.strip(), ent_type))
        if len(deduped) >= max_entities:
            break
    return deduped


def _extract_entities_simple(text: str, max_entities: int = 25) -> list[str]:
    """Extract simple title-cased entities using regex heuristic."""
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    deduped: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        key = c.strip().lower()
        if len(c) < 3 or key in seen:
            continue
        seen.add(key)
        deduped.append(c.strip())
        if len(deduped) >= max_entities:
            break
    return deduped


def _extract_entities_underthesea(text: str, max_entities: int = 25) -> list[tuple[str, str]]:
    try:
        from underthesea import ner as under_ner
    except Exception:
        return [(e, classify_entity_type(e)) for e in _extract_entities_simple(text, max_entities)]

    tags = under_ner(text)
    tagged_tokens = [(token, ner_tag) for token, _pos, _chunk, ner_tag in tags]
    return _collect_bio_entities(tagged_tokens, max_entities)


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
        return [(e, classify_entity_type(e)) for e in _extract_entities_simple(text, max_entities)]

    segmented = segmenter.word_segment(text)
    if not segmented:
        return [(e, classify_entity_type(e)) for e in _extract_entities_simple(text, max_entities)]

    result = model.annotate(text=segmented[0])
    tagged_tokens = [(word, ner_tag) for word, _pos, _chunk, ner_tag in result.get("ner", [])]
    return _collect_bio_entities(tagged_tokens, max_entities)


def extract_entities(text: str, max_entities: int = 25) -> list[tuple[str, str]]:
    """Extract entities and return (name, type) tuples regardless of backend."""
    if settings.ner_backend == "underthesea":
        return _extract_entities_underthesea(text, max_entities)
    if settings.ner_backend == "phonlp":
        return _extract_entities_phonlp(text, max_entities)
    names = _extract_entities_simple(text, max_entities)
    return [(n, classify_entity_type(n)) for n in names]
