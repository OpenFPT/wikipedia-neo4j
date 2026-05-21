"""KG walk extraction and question template engine for ViWiki-MHR dataset."""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.logging_utils import get_logger
from src.neo4j_client import neo4j_client

logger = get_logger(__name__)


@dataclass
class KGWalk:
    """A multi-hop path through the knowledge graph."""

    walk_id: str
    hops: int
    pages: list[str]
    entities: list[dict]
    path_description: str
    evidence_chunks: list[str]


@dataclass
class QAPair:
    """A generated question-answer pair with provenance."""

    qa_id: str
    question: str
    answer: str
    walk_id: str
    hops: int
    question_type: str
    evidence_chunk_ids: list[str]
    source_pages: list[str]


def extract_2hop_walks(limit: int = 5000) -> list[KGWalk]:
    """Extract 2-hop walks: Page1 -[chunk]-> Entity <-[chunk]- Page2."""
    walks: list[KGWalk] = []
    seen: set[str] = set()

    with neo4j_client.session() as session:
        records = session.run(
            """
            MATCH (p1:Page)-[:HAS_CHUNK]->(c1:Chunk)-[:MENTIONS]->(e)<-[:MENTIONS]-(c2:Chunk)<-[:HAS_CHUNK]-(p2:Page)
            WHERE p1 <> p2 AND e.name IS NOT NULL
            WITH p1, p2, e, c1, c2, rand() AS r
            ORDER BY r
            LIMIT $limit
            RETURN p1.title AS page1, p1.url AS url1,
                   p2.title AS page2, p2.url AS url2,
                   e.name AS entity_name, labels(e)[0] AS entity_type,
                   c1.id AS chunk1_id, c1.text AS chunk1_text,
                   c2.id AS chunk2_id, c2.text AS chunk2_text
            """,
            limit=limit,
        )

        for r in records:
            key = f"{r['page1']}|{r['entity_name']}|{r['page2']}"
            if key in seen:
                continue
            seen.add(key)

            walk = KGWalk(
                walk_id=str(uuid.uuid4()),
                hops=2,
                pages=[r["page1"], r["page2"]],
                entities=[{"name": r["entity_name"], "type": r["entity_type"]}],
                path_description=f"{r['page1']} → {r['entity_name']} ({r['entity_type']}) → {r['page2']}",
                evidence_chunks=[r["chunk1_id"], r["chunk2_id"]],
            )
            walks.append(walk)

    logger.info("2-hop walks extracted", extra={"count": len(walks)})
    return walks


def extract_3hop_walks(limit: int = 3000) -> list[KGWalk]:
    """Extract 3-hop walks: Page1 -> Entity1 <- Chunk -> Entity2 <- Page2."""
    walks: list[KGWalk] = []
    seen: set[str] = set()

    with neo4j_client.session() as session:
        records = session.run(
            """
            MATCH (p1:Page)-[:HAS_CHUNK]->(c1:Chunk)-[:MENTIONS]->(e1),
                  (c1)-[:MENTIONS]->(e2),
                  (c2:Chunk)-[:MENTIONS]->(e2)<-[:MENTIONS]-(c3:Chunk)<-[:HAS_CHUNK]-(p2:Page)
            WHERE p1 <> p2 AND e1 <> e2
                  AND e1.name IS NOT NULL AND e2.name IS NOT NULL
            WITH p1, p2, e1, e2, c1, c2, c3, rand() AS r
            ORDER BY r
            LIMIT $limit
            RETURN p1.title AS page1,
                   p2.title AS page2,
                   e1.name AS entity1_name, labels(e1)[0] AS entity1_type,
                   e2.name AS entity2_name, labels(e2)[0] AS entity2_type,
                   c1.id AS chunk1_id, c1.text AS chunk1_text,
                   c3.id AS chunk3_id, c3.text AS chunk3_text
            """,
            limit=limit,
        )

        for r in records:
            key = f"{r['page1']}|{r['entity1_name']}|{r['entity2_name']}|{r['page2']}"
            if key in seen:
                continue
            seen.add(key)

            walk = KGWalk(
                walk_id=str(uuid.uuid4()),
                hops=3,
                pages=[r["page1"], r["page2"]],
                entities=[
                    {"name": r["entity1_name"], "type": r["entity1_type"]},
                    {"name": r["entity2_name"], "type": r["entity2_type"]},
                ],
                path_description=(
                    f"{r['page1']} → {r['entity1_name']} ({r['entity1_type']}) "
                    f"→ {r['entity2_name']} ({r['entity2_type']}) → {r['page2']}"
                ),
                evidence_chunks=[r["chunk1_id"], r["chunk3_id"]],
            )
            walks.append(walk)

    logger.info("3-hop walks extracted", extra={"count": len(walks)})
    return walks


# --- Question Templates ---

_2HOP_TEMPLATES = {
    "Person": [
        "{entity} được đề cập trong bài viết nào ngoài {page1}?",
        "Mối liên hệ giữa {page1} và {page2} thông qua {entity} là gì?",
        "{entity} có vai trò gì trong {page2}?",
    ],
    "Location": [
        "{entity} xuất hiện trong ngữ cảnh nào ở bài {page1} và {page2}?",
        "Địa điểm {entity} liên quan đến {page1} và {page2} như thế nào?",
    ],
    "Organization": [
        "Tổ chức {entity} được nhắc đến trong {page1} và {page2} với vai trò gì?",
        "{entity} có liên quan gì đến cả {page1} và {page2}?",
    ],
    "Work": [
        "Tác phẩm {entity} được đề cập trong {page1} và {page2} như thế nào?",
        "{entity} có ý nghĩa gì trong ngữ cảnh của {page1} và {page2}?",
    ],
}

_3HOP_TEMPLATES = [
    "Mối liên hệ giữa {entity1} và {entity2} trong ngữ cảnh {page1} và {page2} là gì?",
    "Từ {page1}, qua {entity1} và {entity2}, ta có thể tìm thấy thông tin gì ở {page2}?",
    "{entity1} và {entity2} cùng xuất hiện trong bối cảnh nào liên quan đến {page1}?",
]


def _generate_answer_from_chunks(walk: KGWalk, chunks: dict[str, str]) -> str:
    """Build a reference answer from evidence chunks."""
    snippets = []
    for cid in walk.evidence_chunks:
        text = chunks.get(cid, "")
        if text:
            snippets.append(text[:300].strip())
    if not snippets:
        return f"Thông tin liên quan đến {', '.join(e['name'] for e in walk.entities)}."
    return " ".join(snippets)


def generate_qa_from_walks(walks: list[KGWalk]) -> list[QAPair]:
    """Generate QA pairs from KG walks using templates."""
    chunk_ids = set()
    for w in walks:
        chunk_ids.update(w.evidence_chunks)

    chunks: dict[str, str] = {}
    if chunk_ids:
        with neo4j_client.session() as session:
            for batch_start in range(0, len(chunk_ids), 500):
                batch = list(chunk_ids)[batch_start : batch_start + 500]
                records = session.run(
                    "MATCH (c:Chunk) WHERE c.id IN $ids RETURN c.id AS id, c.text AS text",
                    ids=batch,
                )
                for r in records:
                    chunks[r["id"]] = r["text"]

    qa_pairs: list[QAPair] = []

    for walk in walks:
        if walk.hops == 2:
            entity = walk.entities[0]
            etype = entity["type"] or "Person"
            templates = _2HOP_TEMPLATES.get(etype, _2HOP_TEMPLATES["Person"])
            template = random.choice(templates)
            question = template.format(
                entity=entity["name"],
                page1=walk.pages[0],
                page2=walk.pages[1],
            )
            question_type = f"2hop_{etype.lower()}"
        else:
            template = random.choice(_3HOP_TEMPLATES)
            question = template.format(
                entity1=walk.entities[0]["name"],
                entity2=walk.entities[1]["name"],
                page1=walk.pages[0],
                page2=walk.pages[1],
            )
            question_type = "3hop_bridge"

        answer = _generate_answer_from_chunks(walk, chunks)

        qa_pairs.append(QAPair(
            qa_id=str(uuid.uuid4()),
            question=question,
            answer=answer,
            walk_id=walk.walk_id,
            hops=walk.hops,
            question_type=question_type,
            evidence_chunk_ids=walk.evidence_chunks,
            source_pages=walk.pages,
        ))

    logger.info("QA pairs generated", extra={"count": len(qa_pairs)})
    return qa_pairs


def extract_broken_walks(limit: int = 1000) -> list[KGWalk]:
    """Extract broken-link walks for unanswerable questions.

    Entities mentioned in only one page — no cross-page bridge exists.
    """
    walks: list[KGWalk] = []

    with neo4j_client.session() as session:
        records = session.run(
            """
            MATCH (p1:Page)-[:HAS_CHUNK]->(c1:Chunk)-[:MENTIONS]->(e)
            WHERE e.name IS NOT NULL
            WITH e, p1, c1, count{ (c2:Chunk)-[:MENTIONS]->(e) WHERE c2 <> c1 } AS other_mentions
            WHERE other_mentions = 0
            WITH p1, e, c1, rand() AS r
            ORDER BY r
            LIMIT $limit
            RETURN p1.title AS page1,
                   e.name AS entity_name, labels(e)[0] AS entity_type,
                   c1.id AS chunk1_id
            """,
            limit=limit,
        )

        for r in records:
            walk = KGWalk(
                walk_id=str(uuid.uuid4()),
                hops=2,
                pages=[r["page1"], "[NOT_INGESTED]"],
                entities=[{"name": r["entity_name"], "type": r["entity_type"]}],
                path_description=f"{r['page1']} → {r['entity_name']} → [broken link]",
                evidence_chunks=[r["chunk1_id"]],
            )
            walks.append(walk)

    logger.info("Broken walks extracted", extra={"count": len(walks)})
    return walks


_UNANSWERABLE_TEMPLATES = [
    "Theo bài viết {page1}, {entity} có liên quan gì đến {fake_page}?",
    "{entity} được nhắc đến trong {page1}. Bài viết {fake_page} nói gì về {entity}?",
    "Mối quan hệ giữa {entity} trong {page1} và nội dung của {fake_page} là gì?",
]


def generate_unanswerable_qa(walks: list[KGWalk]) -> list[QAPair]:
    """Generate unanswerable questions from broken walks."""
    qa_pairs: list[QAPair] = []

    for walk in walks:
        entity = walk.entities[0]
        fake_page = f"Bài viết không tồn tại về {entity['name']}"
        template = random.choice(_UNANSWERABLE_TEMPLATES)
        question = template.format(
            page1=walk.pages[0],
            entity=entity["name"],
            fake_page=fake_page,
        )

        qa_pairs.append(QAPair(
            qa_id=str(uuid.uuid4()),
            question=question,
            answer="Không thể trả lời. Thông tin này không có trong cơ sở tri thức.",
            walk_id=walk.walk_id,
            hops=walk.hops,
            question_type="unanswerable",
            evidence_chunk_ids=walk.evidence_chunks,
            source_pages=walk.pages,
        ))

    logger.info("Unanswerable QA pairs generated", extra={"count": len(qa_pairs)})
    return qa_pairs


# --- LLM Rewrite Stage ---

_REWRITE_PROMPT = """Viết lại câu hỏi sau đây sao cho tự nhiên hơn, giữ nguyên ý nghĩa.
Chỉ trả về câu hỏi đã viết lại, không giải thích.

Câu hỏi gốc: {question}

Câu hỏi viết lại:"""


def rewrite_questions_with_llm(
    qa_pairs: list[QAPair],
    batch_size: int = 10,
    use_local_model: bool = True,
) -> list[QAPair]:
    """Rewrite template questions using LLM for naturalness.

    Falls back to original question if rewrite fails or is too different.
    """
    if not use_local_model:
        logger.info("LLM rewrite skipped (use_local_model=False)")
        return qa_pairs

    from src.local_llm import chat

    rewritten = 0
    for i in range(0, len(qa_pairs), batch_size):
        batch = qa_pairs[i : i + batch_size]
        for qa in batch:
            if qa.question_type == "unanswerable":
                continue
            try:
                messages = [
                    {"role": "system", "content": "Bạn là trợ lý viết lại câu hỏi tiếng Việt tự nhiên hơn."},
                    {"role": "user", "content": _REWRITE_PROMPT.format(question=qa.question)},
                ]
                result = chat(messages, max_new_tokens=128, temperature=0.3)
                rewritten_q = result.strip().split("\n")[0].strip()

                if _is_valid_rewrite(qa.question, rewritten_q):
                    qa.question = rewritten_q
                    rewritten += 1
            except Exception as e:
                logger.debug("Rewrite failed, keeping original", extra={"error": str(e)})

    logger.info("LLM rewrite complete", extra={"rewritten": rewritten, "total": len(qa_pairs)})
    return qa_pairs


def _is_valid_rewrite(original: str, rewritten: str) -> bool:
    """Check if a rewritten question is valid (not empty, not too different)."""
    if not rewritten or len(rewritten) < 10:
        return False
    if rewritten == original:
        return False
    if len(rewritten) > len(original) * 3:
        return False
    if not rewritten.endswith("?"):
        rewritten += "?"
    return True


# --- QC Pipeline ---


def _check_grounding(qa: QAPair, chunks: dict[str, str]) -> bool:
    """Check if the answer is grounded in evidence chunks."""
    if qa.question_type == "unanswerable":
        return True

    evidence_text = ""
    for cid in qa.evidence_chunk_ids:
        evidence_text += chunks.get(cid, "") + " "

    if not evidence_text.strip():
        return False

    for entity in _extract_key_terms(qa.answer):
        if entity.lower() in evidence_text.lower():
            return True

    return len(qa.answer) > 20


def _extract_key_terms(text: str) -> list[str]:
    """Extract key terms from text for grounding check."""
    import re
    words = re.findall(r"[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*", text)
    return [w for w in words if len(w) > 3]


def _check_well_formed(qa: QAPair) -> bool:
    """Check if the QA pair is well-formed."""
    if not qa.question or len(qa.question) < 15:
        return False
    if not qa.answer or len(qa.answer) < 10:
        return False
    if qa.question == qa.answer:
        return False
    if not any(qa.question.endswith(c) for c in ["?", "？"]):
        return False
    return True


def _check_no_duplicate(qa: QAPair, seen_questions: set[str]) -> bool:
    """Check for near-duplicate questions."""
    normalized = qa.question.lower().strip()
    if normalized in seen_questions:
        return False
    seen_questions.add(normalized)
    return True


def run_qc_pipeline(qa_pairs: list[QAPair]) -> list[QAPair]:
    """Run 3-stage QC: grounding, well-formedness, deduplication."""
    chunk_ids = set()
    for qa in qa_pairs:
        chunk_ids.update(qa.evidence_chunk_ids)

    chunks: dict[str, str] = {}
    if chunk_ids:
        with neo4j_client.session() as session:
            for batch_start in range(0, len(chunk_ids), 500):
                batch = list(chunk_ids)[batch_start : batch_start + 500]
                records = session.run(
                    "MATCH (c:Chunk) WHERE c.id IN $ids RETURN c.id AS id, c.text AS text",
                    ids=batch,
                )
                for r in records:
                    chunks[r["id"]] = r["text"]

    passed: list[QAPair] = []
    seen_questions: set[str] = set()
    rejected = {"grounding": 0, "well_formed": 0, "duplicate": 0}

    for qa in qa_pairs:
        if not _check_well_formed(qa):
            rejected["well_formed"] += 1
            continue
        if not _check_grounding(qa, chunks):
            rejected["grounding"] += 1
            continue
        if not _check_no_duplicate(qa, seen_questions):
            rejected["duplicate"] += 1
            continue
        passed.append(qa)

    logger.info(
        "QC pipeline complete",
        extra={"passed": len(passed), "rejected": rejected, "total": len(qa_pairs)},
    )
    return passed


def save_dataset(qa_pairs: list[QAPair], output_path: str = "data/viwiki_mhr.jsonl") -> None:
    """Save QA pairs to JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            record = {
                "id": qa.qa_id,
                "question": qa.question,
                "answer": qa.answer,
                "metadata": {
                    "walk_id": qa.walk_id,
                    "hops": qa.hops,
                    "question_type": qa.question_type,
                    "evidence_chunk_ids": qa.evidence_chunk_ids,
                    "source_pages": qa.source_pages,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Dataset saved", extra={"path": str(path), "count": len(qa_pairs)})


def generate_dataset(
    two_hop_limit: int = 5000,
    three_hop_limit: int = 3000,
    broken_limit: int = 1000,
    output_path: str = "data/viwiki_mhr.jsonl",
    rewrite: bool = False,
    qc: bool = True,
) -> dict:
    """Run full dataset generation pipeline."""
    logger.info("Starting dataset generation")

    walks_2hop = extract_2hop_walks(limit=two_hop_limit)
    walks_3hop = extract_3hop_walks(limit=three_hop_limit)
    broken_walks = extract_broken_walks(limit=broken_limit)

    qa_answerable = generate_qa_from_walks(walks_2hop + walks_3hop)
    qa_unanswerable = generate_unanswerable_qa(broken_walks)

    all_qa = qa_answerable + qa_unanswerable

    if rewrite:
        all_qa = rewrite_questions_with_llm(all_qa)

    if qc:
        all_qa = run_qc_pipeline(all_qa)

    random.shuffle(all_qa)
    save_dataset(all_qa, output_path)

    stats = {
        "total": len(all_qa),
        "2hop": len(walks_2hop),
        "3hop": len(walks_3hop),
        "unanswerable": len(qa_unanswerable),
        "output_path": output_path,
    }
    logger.info("Dataset generation complete", extra=stats)
    return stats
