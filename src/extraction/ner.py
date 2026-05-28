"""Named Entity Recognition backends and entity classification."""

from __future__ import annotations

import re
import unicodedata

from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

_phonlp_model = None
_phonlp_segmenter = None
_ner_pipeline = None
_videberta_pipeline = None

_NER_TYPE_MAP = {
    "PER": "Person",
    "PERSON": "Person",
    "ORG": "Organization",
    "ORGANIZATION": "Organization",
    "LOC": "Location",
    "LOCATION": "Location",
    "MISC": "Work",
    "MISCELLANEOUS": "Work",
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
    # Abbreviated sports/org forms
    "f.c.", "fc", "sc", "afc", "sfc", "cf",
    "united", "athletic", "academy",
    "đội tuyển", "đội tuyển quốc gia",
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

_KNOWN_GEO_SUFFIXES = {
    "california", "texas", "florida", "new york", "oregon", "nevada",
    "arizona", "ohio", "michigan", "virginia", "illinois", "georgia",
    "hoa kỳ", "việt nam", "pháp", "đức", "anh", "nhật bản",
    "trung quốc", "hàn quốc", "thái lan", "úc", "canada",
    "mexico", "brasil", "ấn độ", "nga", "ý", "tây ban nha",
}

_GEO_PREFIXES = {
    "los", "san", "santa", "santo", "saint", "fort", "port",
    "mount", "lake", "rio", "cape", "new", "el", "la", "las",
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

    # Check org surface patterns before Western-person heuristic
    for pat in _ORG_SURFACE_PATTERNS:
        if pat.search(name):
            return "Organization"

    # Comma-separated "City, State/Country" pattern -> Location
    if "," in name:
        parts = name.split(",", 1)
        suffix = parts[1].strip().lower()
        suffix_words = suffix.split()
        if any(w == kw for kw in _LOCATION_KEYWORDS for w in suffix_words):
            return "Location"
        # Common state/country names after comma
        if suffix in _KNOWN_GEO_SUFFIXES:
            return "Location"

    # Western person: 2-4 capitalized words, no keyword match, no comma, no geo-prefix
    if "," not in name and 2 <= len(words) <= 4 and all(w[0].isupper() for w in name.split() if w):
        if words[0].lower() in _GEO_PREFIXES:
            return "Location"
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
        logger.warning("underthesea import failed, falling back to simple NER backend")
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
        logger.warning("phonlp/vncorenlp load failed, falling back to simple NER backend")
        return [(e, classify_entity_type(e)) for e in _extract_entities_simple(text, max_entities)]

    segmented = segmenter.word_segment(text)
    if not segmented:
        return [(e, classify_entity_type(e)) for e in _extract_entities_simple(text, max_entities)]

    result = model.annotate(text=segmented[0])
    tagged_tokens = [(word, ner_tag) for word, _pos, _chunk, ner_tag in result.get("ner", [])]
    return _collect_bio_entities(tagged_tokens, max_entities)


def _get_ner_pipeline():
    """Lazy-load HuggingFace token-classification pipeline."""
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline as hf_pipeline

        _ner_pipeline = hf_pipeline(
            "token-classification",
            model=settings.ner_model_id,
            aggregation_strategy="simple",
            device=-1,
            truncation=True,
        )
        logger.info("Loaded NER model: %s", settings.ner_model_id)
    return _ner_pipeline


def normalize_entity(name: str) -> str:
    """Normalize entity name: NFKC, strip punctuation, collapse whitespace."""
    name = unicodedata.normalize("NFKC", name)
    name = name.strip(" .,;:!?\"'[]{}«»")
    name = " ".join(name.split())
    return name


_DISAMBIG_LOCATION_HINTS = [
    "thành phố", "tỉnh", "quận", "huyện", "xã", "phường",
    "sông", "núi", "hồ", "biển", "đảo", "vịnh", "bán đảo",
    "cao nguyên", "đồng bằng", "tiểu bang", "bang", "quốc gia",
    "lãnh thổ", "vùng", "châu", "khu vực", "thị xã", "thị trấn",
    "city", "state", "province", "country", "river", "mountain",
    "lake", "island", "district", "county", "region", "peninsula",
]
_DISAMBIG_ORG_HINTS = [
    "tổ chức", "công ty", "đảng", "đội bóng", "câu lạc bộ",
    "trường", "đại học", "viện", "hãng", "tập đoàn",
    "organization", "company", "party", "club", "university",
]
_DISAMBIG_PERSON_HINTS = [
    "nhà văn", "nhà thơ", "chính trị gia", "vua", "hoàng đế",
    "tướng", "ca sĩ", "diễn viên", "nhạc sĩ", "họa sĩ",
    "writer", "politician", "singer", "actor", "composer",
]
_DISAMBIG_WORK_HINTS = [
    "phim", "tiểu thuyết", "bài hát", "album", "truyện",
    "ca khúc", "sách", "trò chơi", "chương trình",
    "film", "movie", "novel", "song", "album", "book", "game", "series",
]

_DISAMBIG_TYPE_MAP: dict[str, str] = {}
for _hint in _DISAMBIG_LOCATION_HINTS:
    _DISAMBIG_TYPE_MAP[_hint] = "Location"
for _hint in _DISAMBIG_ORG_HINTS:
    _DISAMBIG_TYPE_MAP[_hint] = "Organization"
for _hint in _DISAMBIG_PERSON_HINTS:
    _DISAMBIG_TYPE_MAP[_hint] = "Person"
for _hint in _DISAMBIG_WORK_HINTS:
    _DISAMBIG_TYPE_MAP[_hint] = "Work"


def strip_disambiguation(name: str) -> tuple[str, str]:
    """Strip parenthetical disambiguation suffix and return (clean_name, hint).

    "Eureka (word)" -> ("Eureka", "word")
    "Sacramento, California" -> ("Sacramento, California", "")
    """
    match = re.search(r"\s*\(([^)]+)\)\s*$", name)
    if match:
        hint = match.group(1).strip()
        clean = name[: match.start()].strip()
        return (clean, hint) if clean else (name, "")
    match_open = re.search(r"\s*\([^)]*$", name)
    if match_open:
        clean = name[: match_open.start()].strip()
        hint = name[match_open.start():].strip(" (")
        return (clean, hint) if clean else (name, "")
    return (name, "")


def classify_disambiguation_hint(hint: str) -> str:
    """Classify entity type from a disambiguation hint string."""
    if not hint:
        return "Unknown"
    hint_lower = hint.lower()
    for key, etype in _DISAMBIG_TYPE_MAP.items():
        if key in hint_lower:
            return etype
    return "Unknown"


_PERSON_CONTEXT_KW = [
    "ông", "bà", "chủ tịch", "tổng thống", "thủ tướng",
    "giáo sư", "tiến sĩ", "nghệ sĩ", "nhà văn", "tướng",
]
_LOCATION_CONTEXT_KW = [
    "thành phố", "tỉnh", "quận", "huyện", "tại", "ở", "đến", "từ",
]


def disambiguate_type_by_context(
    name: str, context: str, bio_type: str
) -> str:
    """Use surrounding text to resolve ambiguous entity types."""
    if bio_type not in ("Unknown", "Location", "Person"):
        return bio_type
    ctx_lower = context.lower()
    name_pos = ctx_lower.find(name.lower())
    prefix = ctx_lower[max(0, name_pos - 30):name_pos] if name_pos > 0 else ""

    for kw in _LOCATION_CONTEXT_KW:
        if kw in prefix:
            return "Location"
    for kw in _PERSON_CONTEXT_KW:
        if kw in prefix:
            return "Person"
    return bio_type


def _extract_entities_phobert(
    text: str, max_entities: int = 25
) -> list[tuple[str, str]]:
    """Extract entities using HuggingFace transformer pipeline with confidence filtering."""
    try:
        pipe = _get_ner_pipeline()
    except Exception:
        logger.warning("Failed to load NER pipeline, falling back to simple backend")
        return [(e, classify_entity_type(e)) for e in _extract_entities_simple(text, max_entities)]

    truncated = text[:2048]
    raw_entities = pipe(truncated)

    threshold = settings.ner_confidence_threshold
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()

    for ent in raw_entities:
        score = ent.get("score", 0.0)
        if score < threshold:
            continue
        word = ent.get("word", "").replace("##", "")
        word = normalize_entity(word)
        if not word or len(word) < 2:
            continue

        raw_label = ent.get("entity_group", ent.get("entity", ""))
        mapped_type = _NER_TYPE_MAP.get(raw_label, classify_entity_type(word))

        mapped_type = disambiguate_type_by_context(word, text, mapped_type)

        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((word, mapped_type))
        if len(deduped) >= max_entities:
            break

    return deduped


def _get_videberta_pipeline():
    """Lazy-load HuggingFace token-classification pipeline for ViDeBERTa."""
    global _videberta_pipeline
    if _videberta_pipeline is None:
        from transformers import AutoTokenizer, pipeline as hf_pipeline

        tokenizer = AutoTokenizer.from_pretrained(
            settings.videberta_model_id, model_max_length=512
        )
        _videberta_pipeline = hf_pipeline(
            "token-classification",
            model=settings.videberta_model_id,
            tokenizer=tokenizer,
            aggregation_strategy="none",
            device=-1,
        )
        logger.info("Loaded ViDeBERTa NER model: %s", settings.videberta_model_id)
    return _videberta_pipeline


def _extract_entities_videberta(
    text: str, max_entities: int = 25
) -> list[tuple[str, str]]:
    """Extract entities using ViDeBERTa token-classification with BIO accumulation."""
    try:
        pipe = _get_videberta_pipeline()
    except Exception:
        logger.warning("Failed to load ViDeBERTa pipeline, falling back to simple backend")
        return [(e, classify_entity_type(e)) for e in _extract_entities_simple(text, max_entities)]

    truncated = text[:2048]
    raw_tokens = pipe(truncated)

    # Build (token, ner_tag) pairs for BIO accumulation
    tagged_tokens: list[tuple[str, str]] = []
    for tok in raw_tokens:
        word = tok.get("word", "").replace("▁", " ").strip()
        if not word:
            continue
        label = tok.get("entity", "O")
        tagged_tokens.append((word, label))

    entities = _collect_bio_entities(tagged_tokens, max_entities)

    # Refine types using keyword heuristics
    refined: list[tuple[str, str]] = []
    for name, bio_type in entities:
        name = normalize_entity(name)
        if not name or len(name) < 2:
            continue
        if bio_type == "Unknown":
            refined.append((name, classify_entity_type(name)))
        else:
            kw_type = classify_entity_type(name)
            if kw_type == "Organization":
                refined.append((name, "Organization"))
            else:
                refined.append((name, bio_type))

    return refined


def extract_entities_batch(
    texts: list[str], max_entities: int = 25
) -> list[list[tuple[str, str]]]:
    """Batch extract entities from multiple texts (optimized for phobert backend)."""
    if settings.ner_backend != "phobert":
        return [extract_entities(t, max_entities) for t in texts]

    try:
        pipe = _get_ner_pipeline()
    except Exception:
        logger.warning("phobert batch pipeline load failed, falling back to per-text extraction")
        return [extract_entities(t, max_entities) for t in texts]

    truncated = [t[:2048] for t in texts]
    all_raw = pipe(truncated, batch_size=32)

    threshold = settings.ner_confidence_threshold
    results: list[list[tuple[str, str]]] = []

    for i, raw_entities in enumerate(all_raw):
        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for ent in raw_entities:
            score = ent.get("score", 0.0)
            if score < threshold:
                continue
            word = ent.get("word", "").replace("##", "")
            word = normalize_entity(word)
            if not word or len(word) < 2:
                continue

            raw_label = ent.get("entity_group", ent.get("entity", ""))
            mapped_type = _NER_TYPE_MAP.get(raw_label, classify_entity_type(word))
            mapped_type = disambiguate_type_by_context(word, texts[i], mapped_type)

            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append((word, mapped_type))
            if len(deduped) >= max_entities:
                break
        results.append(deduped)

    return results


def _extract_entities_wikilink(
    raw_text: str, max_entities: int = 500
) -> list[tuple[str, str]]:
    """Extract entities from [[wiki links]] in raw Wikipedia markup."""
    from src.text_utils import extract_wikilinks

    links = extract_wikilinks(raw_text)
    results: list[tuple[str, str]] = []
    for target, _display in links:
        clean_name, hint = strip_disambiguation(target)
        if not clean_name:
            continue
        hint_type = classify_disambiguation_hint(hint)
        if hint_type != "Unknown":
            entity_type = hint_type
        else:
            entity_type = classify_entity_type(clean_name)
        results.append((clean_name, entity_type))
        if len(results) >= max_entities:
            break
    return results


_NOISE_PATTERNS = [
    re.compile(r"^[,.\-;:\s]"),
    re.compile(r"^(năm|mùa giải|vòng|thế kỷ|tháng)\b", re.IGNORECASE),
    re.compile(r"^\d{4}$"),
    re.compile(r"^(áo số|trận đấu|bàn thắng)\b"),
    re.compile(r"^(sự kiện|khoảnh khắc|việc|bối cảnh|bên cạnh|ngày)\b"),
    re.compile(r"^(trước|sau|trong|ngoài|khoảng)\b"),
    re.compile(r"^(chứng|trực chứng|khai mở|kinh điển)\b"),
    re.compile(r"\d+\s*(tháng|năm|tuổi|km|m²)"),
    # Wikilink-specific navigation/category noise
    re.compile(r"^(Danh sách|Lịch sử|Tham khảo|Xem thêm|Chú thích)\b"),
    re.compile(r"^\d+[\s\-]*[–\-]?\s*\d*$"),
    re.compile(r"^(và|của|các|những|được|này|đó|hay|hoặc|cũng)$", re.IGNORECASE),
]

_ORG_SURFACE_PATTERNS = [
    re.compile(r"\b(F\.?C\.?|United|City|FC|SC|CF|AFC|SFC)\b", re.IGNORECASE),
    re.compile(r"\b(Club|Team|Academy|Athletic)\b", re.IGNORECASE),
    re.compile(r"\b(Inc|Ltd|Corp|Co\.|LLC|GmbH|S\.A\.)\b", re.IGNORECASE),
]


def _is_noise(name: str) -> bool:
    """Return True if the entity name is likely not a real entity."""
    if len(name) <= 1:
        return True
    if len(name.split()) == 1 and len(name) <= 2:
        return True
    if name[0].islower() and not any(c.isupper() for c in name[1:]):
        return True
    for pat in _NOISE_PATTERNS:
        if pat.search(name):
            return True
    return False


def _reclassify_type(name: str, current_type: str) -> str:
    """Override type when surface patterns strongly indicate a different type."""
    for pat in _ORG_SURFACE_PATTERNS:
        if pat.search(name):
            return "Organization"
    return current_type


def postprocess_entities(
    entities: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Filter noise and fix entity types after extraction."""
    result: list[tuple[str, str]] = []
    seen: set[str] = set()

    for name, etype in entities:
        name = normalize_entity(name)
        if not name or _is_noise(name):
            continue

        etype = _reclassify_type(name, etype)

        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append((name, etype))

    return result


def extract_entities(text: str, max_entities: int = 25) -> list[tuple[str, str]]:
    """Extract entities and return (name, type) tuples regardless of backend."""
    if settings.ner_backend == "wikilink":
        raw = _extract_entities_wikilink(text, max_entities=500)
        return postprocess_entities(raw)
    if settings.ner_backend == "phobert":
        raw = _extract_entities_phobert(text, max_entities)
    elif settings.ner_backend == "videberta":
        raw = _extract_entities_videberta(text, max_entities)
    elif settings.ner_backend == "underthesea":
        raw = _extract_entities_underthesea(text, max_entities)
    elif settings.ner_backend == "phonlp":
        raw = _extract_entities_phonlp(text, max_entities)
    else:
        names = _extract_entities_simple(text, max_entities)
        raw = [(n, classify_entity_type(n)) for n in names]
    return postprocess_entities(raw)
