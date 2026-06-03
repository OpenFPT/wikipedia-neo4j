"""Multi-hop question decomposition and answer synthesis."""

from __future__ import annotations

from src.logging_utils import get_logger
from src.prompts import DECOMPOSE_QUESTION_PROMPT, SYNTHESIS_PROMPT
from src.retrieval.fusion import QueryResult

from src.orchestration._parsing import parse_agent_response

logger = get_logger(__name__)


def decompose_question(question: str) -> list[str] | None:
    """Decompose a complex question into sub-questions."""
    try:
        from src.infrastructure.local_llm import chat

        prompt = DECOMPOSE_QUESTION_PROMPT.format(question=question)
        response = chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=256,
            temperature=0.1,
        )

        parsed = parse_agent_response(response)
        if parsed and "sub_questions" in parsed:
            sub_qs = parsed["sub_questions"]
            if isinstance(sub_qs, list) and len(sub_qs) <= 4:
                questions = [sq.get("question", "") for sq in sub_qs if sq.get("question")]
                if questions:
                    logger.info("Question decomposed", extra={"count": len(questions)})
                    return questions

    except Exception as e:
        logger.warning(f"Question decomposition failed: {e}")

    return None


def synthesize_answers(
    original_question: str,
    sub_qa_pairs: list[tuple[str, str]],
) -> str:
    """Synthesize answers from sub-questions into a final answer."""
    try:
        from src.infrastructure.local_llm import chat

        sub_qa_text = "\n".join(
            [f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(sub_qa_pairs)]
        )

        prompt = SYNTHESIS_PROMPT.format(
            question=original_question,
            sub_qa=sub_qa_text,
        )

        response = chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=512,
            temperature=0.1,
        )

        return response.strip()

    except Exception as e:
        logger.warning(f"Answer synthesis failed: {e}")
        return ""


def agent_query_with_decomposition(question: str, sub_questions: list[str]) -> QueryResult:
    """Answer a complex question by solving sub-questions sequentially."""
    from src.orchestration.agent_loop import agent_query_standard

    sub_qa_pairs: list[tuple[str, str]] = []
    all_citations: list[dict] = []
    seen_chunk_ids: set[str] = set()

    for i, sub_q in enumerate(sub_questions):
        logger.info(f"Answering sub-question {i+1}/{len(sub_questions)}")

        # Answer each sub-question using standard agent loop
        result = agent_query_standard(sub_q, top_k=4)
        sub_qa_pairs.append((sub_q, result.answer))

        # Collect citations
        for citation in result.citations:
            cid = citation.get("chunk_id", "")
            if cid and cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                all_citations.append(citation)

    # Synthesize final answer from sub-answers
    final_answer = synthesize_answers(question, sub_qa_pairs)

    if not final_answer:
        # Fallback if synthesis fails
        final_answer = "Không thể tổng hợp câu trả lời từ các câu hỏi con."

    logger.info("Decomposition-based query completed", extra={"sub_questions": len(sub_questions)})
    return QueryResult(answer=final_answer, citations=all_citations)
