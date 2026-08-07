"""Rule-based query planner for GraphRAG retrieval.

Goal: reduce "can't answer" cases caused by brittle free-form Cypher generation.
We classify common Vietnamese question intents and route to stable Cypher templates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    intent: str
    subject: str | None = None


_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = _WS_RE.sub(" ", s)
    return s


def _extract_subject_simple(q: str) -> str | None:
    # Heuristic: subject is often the prefix before "là ai/là gì/khi nào/ở đâu/..." in Vietnamese.
    qn = _normalize(q)
    # Special-case "Ai ...": subject is often the remainder after "ai".
    if qn.startswith("ai "):
        rest = qn[3:].strip()
        return rest or None
    for sep in (
        " là ai",
        " la ai",
        " là gì",
        " la gi",
        " khi nào",
        " khi nao",
        " ở đâu",
        " o dau",
        " tại đâu",
        " tai dau",
        " vì ",
        " vi ",
        " theo ",
        " do ",
    ):
        if sep in qn:
            left = qn.split(sep, 1)[0].strip()
            return left or None
    return None


def plan(question: str) -> Plan:
    qn = _normalize(question)

    # Treaty/decision "which one" questions.
    if any(k in qn for k in ("hiệp định nào", "hiep dinh nao", "hiệp ước nào", "hiep uoc nao")):
        return Plan(intent="which_treaty", subject=_extract_subject_simple(question))

    # Definition questions.
    if any(k in qn for k in ("là ai", "la ai", "là gì", "la gi")):
        return Plan(intent="definition", subject=_extract_subject_simple(question))

    # Participant questions (who signed/attended/founded/led...).
    if "ai" in qn and any(
        k in qn
        for k in (
            "hội nghị",
            "hoi nghi",
            "hội thảo",
            "hoi thao",
            "tham dự",
            "tham gia",
            "ký",
            "ki",
            "kí",
            "sáng lập",
            "thành lập",
            "lanh dao",
            "lãnh đạo",
            "đứng đầu",
        )
    ):
        return Plan(intent="participants", subject=_extract_subject_simple(question))

    # Legal/decision reference questions ("theo nghị định/quyết định/luật nào").
    if (
        any(k in qn for k in ("theo ", "căn cứ", "can cu", "quyết định nào", "quyet dinh nao", "nghị định nào", "nghi dinh nao"))
        and any(
            k in qn
            for k in (
                "nghị định",
                "nghi dinh",
                "quyết định",
                "quyet dinh",
                "luật",
                "luat",
                "thông tư",
                "thong tu",
            )
        )
    ):
        # If user asks "số nào" (decision number), prefer the stricter regex template.
        if any(k in qn for k in ("số nào", "so nao", "number nào", "so bao nhieu")):
            return Plan(intent="which_decision_regex", subject=_extract_subject_simple(question))
        return Plan(intent="which_decision", subject=_extract_subject_simple(question))

    # List/enumeration questions.
    if any(qn.startswith(k) for k in ("liệt kê", "liet ke", "kể tên", "ke ten", "danh sách", "danh sach")):
        return Plan(intent="list", subject=_extract_subject_simple(question))

    # "What/which details" questions.
    if any(k in qn for k in ("gì", "gi", "như thế nào", "nhu the nao", "ra sao")):
        return Plan(intent="details", subject=_extract_subject_simple(question))

    # When/where.
    if any(k in qn for k in ("khi nào", "khi nao", "năm nào", "nam nao", "ngày nào", "ngay nao")):
        return Plan(intent="when", subject=_extract_subject_simple(question))
    if any(k in qn for k in ("ở đâu", "o dau", "tại đâu", "tai dau")):
        return Plan(intent="where", subject=_extract_subject_simple(question))

    # Why / because.
    if any(k in qn for k in ("vì ", "vi ", "do ", "bởi ", "boi ")):
        return Plan(intent="why", subject=_extract_subject_simple(question))

    return Plan(intent="unknown", subject=_extract_subject_simple(question))
