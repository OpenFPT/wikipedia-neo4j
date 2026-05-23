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
    "institute", "foundation", "association", "organization",
    "committee", "council", "agency", "bureau", "commission",
    "force", "society", "league", "federation", "union", "party",
    "group", "team", "club",
    # Vietnamese
    "tập đoàn", "công ty", "ngân hàng", "đại học", "viện",
    "trường", "học viện", "bộ", "sở", "ủy ban",
    "hội", "đảng", "liên đoàn", "hiệp hội", "tổ chức",
    "quỹ", "ban", "cục", "vụ", "tổng cục",
    "đoàn", "đội", "câu lạc bộ", "liên minh",
    "quốc hội", "chính phủ", "nhà nước",
]

_LOCATION_KEYWORDS = [
    "city", "province", "district", "county", "state",
    "river", "mount", "mountain", "lake", "sea", "island", "bay",
    "country", "continent", "region", "area", "village", "town",
    # Vietnamese
    "thành phố", "tỉnh", "quận", "huyện", "xã", "phường",
    "sông", "núi", "hồ", "biển", "đảo", "vịnh",
    "miền", "vùng", "châu", "lục địa",
    "thị xã", "thị trấn", "đường", "phố",
    "bán đảo", "cao nguyên", "đồng bằng", "dãy núi",
]

_WORK_KEYWORDS = [
    "film", "movie", "novel", "book", "album", "song",
    "series", "show", "opera", "symphony", "painting",
    # Vietnamese
    "phim", "tiểu thuyết", "tác phẩm", "bài hát",
    "truyện", "kịch", "vở", "bản giao hưởng",
    "cuốn sách", "album", "ca khúc",
]

_VN_FAMILY_NAMES = {
    "nguyễn", "trần", "lê", "phạm", "hoàng", "huỳnh",
    "phan", "vũ", "võ", "đặng", "bùi", "đỗ", "hồ",
    "ngô", "dương", "lý", "lương", "trương", "đinh",
    "tô", "mai", "tạ", "triệu", "đào", "lâm", "cao",
    "hà", "tăng", "châu", "quách", "thái", "diệp",
}


def classify_entity_type(name: str) -> str:
    """Classify an entity name into Person/Organization/Location/Work/Unknown.

    Uses word-boundary-aware matching to avoid false positives from substrings.
    """
    lowered = name.lower()
    words = lowered.split()

    # Check org keywords (prefix/contains for multi-word keywords)
    for kw in _ORG_KEYWORDS:
        if " " in kw:
            if kw in lowered:
                return "Organization"
        elif lowered.startswith(kw + " ") or lowered.startswith(kw + "_"):
            return "Organization"
        elif any(w == kw for w in words):
            return "Organization"

    # Check location keywords
    for kw in _LOCATION_KEYWORDS:
        if " " in kw:
            if kw in lowered:
                return "Location"
        elif lowered.startswith(kw + " ") or lowered.startswith(kw + "_"):
            return "Location"
        elif any(w == kw for w in words[1:]):
            # Only match non-first word to avoid "Hồ Chí Minh" matching "hồ"
            return "Location"

    # Check work keywords
    for kw in _WORK_KEYWORDS:
        if kw in lowered:
            return "Work"

    # Vietnamese person: first word is a known family name AND has 2-4 words
    if len(words) >= 2 and words[0] in _VN_FAMILY_NAMES:
        return "Person"

    # Western person: 2-4 capitalized words, no keyword match
    if 2 <= len(words) <= 4 and all(w[0].isupper() for w in name.split() if w):
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
    entities = _collect_bio_entities(tagged_tokens, max_entities)
    refined = []
    for name, bio_type in entities:
        if bio_type == "Unknown":
            refined.append((name, classify_entity_type(name)))
        else:
            kw_type = classify_entity_type(name)
            if kw_type == "Organization":
                refined.append((name, "Organization"))
            elif kw_type == "Person" and len(name.split()) >= 3 and name.lower().split()[0] in _VN_FAMILY_NAMES:
                # High-confidence person: starts with Vietnamese family name
                refined.append((name, "Person"))
            else:
                refined.append((name, bio_type))
    return refined


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
