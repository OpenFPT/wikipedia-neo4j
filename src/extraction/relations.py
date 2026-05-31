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

EXTRACTION_PROMPT = """Extract relationships between entities from the following Vietnamese text.

Only extract relationships of these types:
- FOUNDED_BY: Organization was founded by Person
- LOCATED_IN: Entity is located in Location
- BORN_IN: Person was born in Location
- MEMBER_OF: Person is a member of Organization
- PART_OF: Entity is part of another Entity
- CREATED_BY: Work/Entity was created by Person/Organization

Return a JSON array of triples:
[{{"subject": "entity1", "relation": "RELATION_TYPE", "object": "entity2"}}]

If no relationships can be extracted, return an empty array: []

Text: {text}"""


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
        from src.config import settings
        from src.infrastructure.llm import _client_pool

        from google.genai import types

        clients = _client_pool()
        client = clients[0]
        resp = client.models.generate_content(
            model=settings.gemini_model_text,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )
        raw = resp.text or "[]"

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
        if relation not in RELATION_TYPES:
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
