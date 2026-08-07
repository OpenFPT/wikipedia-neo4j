"""Cypher template library for common Vietnamese question intents.

All templates MUST return the shape expected by the API layer:
  page_title, page_url, page_id, chunk_id, chunk_text, score
"""

from __future__ import annotations

import re


def _escape_lucene_term(s: str) -> str:
    # Minimal escaping for Neo4j fulltext query syntax.
    # We keep it simple to avoid breaking queries; fallback is quoted phrase.
    s = (s or "").strip()
    if not s:
        return ""
    if re.search(r'[\+\-\!\(\)\{\}\[\]\^"~\*\?:\\\/]|&&|\|\|', s):
        s = s.replace('"', '\\"')
        return f"\"{s}\""
    return s


def build_fulltext_query(question: str, subject: str | None = None) -> str:
    q = (question or "").strip()
    if subject:
        # Weighted: subject is usually the anchor.
        return f"{_escape_lucene_term(subject)} {q}"
    return q


TEMPLATE_DEFINITION = """
CALL db.index.fulltext.queryNodes('page_title_ft', $q) YIELD node, score
MATCH (node:Page)-[:HAS_CHUNK]->(c:Chunk)
RETURN node.title AS page_title,
       node.url AS page_url,
       node.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS score
ORDER BY score DESC, c.sequence_number ASC
LIMIT $top_k
"""


TEMPLATE_WHEN = """
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node AS c, score
MATCH (p:Page)-[:HAS_CHUNK]->(c)
WHERE c.text =~ '(?is).*\\b(ngày|tháng|năm|\\d{4})\\b.*'
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS score
ORDER BY score DESC
LIMIT $top_k
"""


TEMPLATE_WHERE = """
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node AS c, score
MATCH (p:Page)-[:HAS_CHUNK]->(c)
WHERE c.text =~ '(?is).*(tại|ở|thuộc|đặt tại|đóng tại|thành phố|tỉnh|huyện|xã|quận|phường).*'
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS score
ORDER BY score DESC
LIMIT $top_k
"""


TEMPLATE_WHY = """
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node AS c, score
MATCH (p:Page)-[:HAS_CHUNK]->(c)
WHERE c.text =~ '(?is).*(vì|do|bởi|nhằm|để).*'
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS score
ORDER BY score DESC
LIMIT $top_k
"""


TEMPLATE_WHICH_TREATY = """
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node AS c, score
MATCH (p:Page)-[:HAS_CHUNK]->(c)
WHERE c.text =~ '(?is).*(hiệp định|hiệp ước).*'
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS score
ORDER BY score DESC
LIMIT $top_k
"""

TEMPLATE_DETAILS = """
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node AS c, score
MATCH (p:Page)-[:HAS_CHUNK]->(c)
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS score
ORDER BY score DESC
LIMIT $top_k
"""

TEMPLATE_PARTICIPANTS = """
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node AS c, score
MATCH (p:Page)-[:HAS_CHUNK]->(c)
WHERE c.text =~ '(?is).*(tham dự|tham gia|ký|kí|sáng lập|thành lập|lãnh đạo|đứng đầu|đại biểu|phái đoàn).*'
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS score
ORDER BY score DESC
LIMIT $top_k
"""

TEMPLATE_WHICH_DECISION = """
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node AS c, score
MATCH (p:Page)-[:HAS_CHUNK]->(c)
WHERE c.text =~ '(?is).*(nghị định|quyết định|luật|thông tư).*'
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS score
ORDER BY score DESC
LIMIT $top_k
"""

TEMPLATE_WHICH_DECISION_REGEX = """
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node AS c, score
MATCH (p:Page)-[:HAS_CHUNK]->(c)
WHERE c.text =~ '(?is).*(quyết định\\s+số|nghị định\\s+số|nghị quyết\\s+số|thông tư\\s+số).*'
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS score
ORDER BY score DESC
LIMIT $top_k
"""

TEMPLATE_LIST = """
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node AS c, score
MATCH (p:Page)-[:HAS_CHUNK]->(c)
RETURN p.title AS page_title,
       p.url AS page_url,
       p.id AS page_id,
       c.id AS chunk_id,
       c.text AS chunk_text,
       score AS score
ORDER BY score DESC
LIMIT $top_k
"""


def pick_template(intent: str) -> str | None:
    return {
        "definition": TEMPLATE_DEFINITION,
        "when": TEMPLATE_WHEN,
        "where": TEMPLATE_WHERE,
        "why": TEMPLATE_WHY,
        "which_treaty": TEMPLATE_WHICH_TREATY,
        "details": TEMPLATE_DETAILS,
        "participants": TEMPLATE_PARTICIPANTS,
        "which_decision": TEMPLATE_WHICH_DECISION,
        "which_decision_regex": TEMPLATE_WHICH_DECISION_REGEX,
        "list": TEMPLATE_LIST,
    }.get(intent)
