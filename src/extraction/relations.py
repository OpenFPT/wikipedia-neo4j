"""LLM-based relation extraction with typed ontology."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.logging_utils import get_logger

logger = get_logger(__name__)

RELATION_TYPES = [
    "FOUNDED_BY",
    "LOCATED_IN",
    "BORN_IN",
    "MEMBER_OF",
    "PART_OF",
    "CREATED_BY",
]

EXTENDED_RELATION_TYPES = [
    "DIED_IN",
    "NATIONALITY",
    "OCCUPATION",
    "CAPITAL_OF",
    "LEADER_OF",
    "WORKED_AT",
    "STUDIED_AT",
    "AWARDED",
    "SPOUSE_OF",
    "PARENT_OF",
]

_ALLOWED_RELATION_TYPES = set(RELATION_TYPES + EXTENDED_RELATION_TYPES)

EXTRACTION_PROMPT = """Trích xuất các mối quan hệ giữa các thực thể từ văn bản Wikipedia tiếng Việt dưới đây.

Các loại quan hệ được phép:
- FOUNDED_BY: Tổ chức được thành lập bởi Người (vd: "Công ty X được thành lập bởi Y")
- LOCATED_IN: Thực thể nằm ở Địa điểm (vd: "X nằm ở tỉnh Y")
- BORN_IN: Người sinh ra ở Địa điểm (vd: "X sinh tại Y")
- DIED_IN: Người mất ở Địa điểm (vd: "X mất tại Y")
- MEMBER_OF: Người là thành viên của Tổ chức (vd: "X là thành viên của Y")
- PART_OF: Thực thể là một phần của Thực thể khác (vd: "X là một phần của Y")
- CREATED_BY: Tác phẩm/Thực thể được tạo bởi Người/Tổ chức (vd: "X được viết bởi Y")
- NATIONALITY: Người có quốc tịch (vd: "X là người Việt Nam")
- OCCUPATION: Người có nghề nghiệp (vd: "X là nhà văn/chính trị gia")
- CAPITAL_OF: Địa điểm là thủ đô/trung tâm của Thực thể (vd: "Hà Nội là thủ đô của Việt Nam")
- LEADER_OF: Người lãnh đạo Tổ chức/Quốc gia (vd: "X là chủ tịch của Y")
- WORKED_AT: Người làm việc tại Tổ chức (vd: "X làm việc tại Y")
- STUDIED_AT: Người học tại Tổ chức (vd: "X học tại trường Y")
- AWARDED: Người/Thực thể nhận giải thưởng (vd: "X nhận giải Y")
- SPOUSE_OF: Người là vợ/chồng của Người (vd: "X kết hôn với Y")
- PARENT_OF: Người là cha/mẹ của Người (vd: "X là cha của Y")

Trả về JSON array các triple. Trích xuất TẤT CẢ quan hệ tìm thấy, kể cả khi không chắc chắn hoàn toàn:
[{{"subject": "tên thực thể 1", "relation": "LOẠI_QUAN_HỆ", "object": "tên thực thể 2"}}]

Nếu không tìm thấy quan hệ nào, trả về: []

Văn bản: {text}"""


@dataclass
class Triple:
    """An extracted relation triple."""

    subject: str
    relation: str
    object: str
    confidence: float = 1.0


def extract_relations(text: str, use_local: bool = True) -> list[Triple]:
    """Extract relation triples from text using LLM.

    Args:
        text: Input text (Vietnamese)
        use_local: If True, use local model; otherwise use Gemini

    Returns:
        List of extracted Triple objects
    """
    prompt = EXTRACTION_PROMPT.format(text=text[:2000])  # Cap input length

    if use_local:
        from src.infrastructure.local_llm import chat

        messages = [
            {"role": "system", "content": "You are a relation extraction system. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        raw = chat(messages, max_new_tokens=512, temperature=0.1)
    else:
        from openai import OpenAI
        from src.config import settings

        client = OpenAI(
            api_key=settings.anthropic_api_key,
            base_url=settings.cn_base_url + "/v1",
        )
        resp = client.chat.completions.create(
            model=settings.cn_model_text,
            max_tokens=2048,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a relation extraction system. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or "[]"

    return _parse_triples(raw)


def _parse_triples(raw: str) -> list[Triple]:
    """Parse LLM output into Triple objects."""
    # Strip code fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    # Find JSON array
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return []

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning("Failed to parse relation extraction output")
        return []

    triples = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subject = item.get("subject", "").strip()
        relation = item.get("relation", "").strip().upper()
        obj = item.get("object", "").strip()

        if not subject or not obj:
            continue
        if relation not in _ALLOWED_RELATION_TYPES:
            continue

        triples.append(
            Triple(
                subject=subject,
                relation=relation,
                object=obj,
                confidence=item.get("confidence", 1.0),
            )
        )

    return triples


def extract_relations_batch(texts: list[str], use_local: bool = True) -> list[list[Triple]]:
    """Extract relations from multiple texts."""
    results = []
    for text in texts:
        try:
            triples = extract_relations(text, use_local=use_local)
            results.append(triples)
        except Exception as e:
            logger.warning("Relation extraction failed for text: %s", e)
            results.append([])
    return results
