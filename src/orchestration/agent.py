"""Backward-compatible ReAct agent API.

This shim preserves the historical `src.orchestration.agent` surface while the
implementation now lives in smaller submodules.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import settings
from src.infrastructure.neo4j_client import neo4j_client  # noqa: F401
from src.orchestration import complexity as _complexity_mod
from src.orchestration import decomposition as _decomposition_mod
from src.orchestration import voting as _voting_mod
from src.orchestration._parsing import parse_agent_response as parse_agent_response
from src.orchestration.agent_loop import (
    MAX_ITERATIONS as MAX_ITERATIONS,
    SYSTEM_PROMPT as SYSTEM_PROMPT,
    TOOLS as TOOLS,
    _check_sufficiency as _check_sufficiency,
    _tool_get_passage as _tool_get_passage,
    _tool_kg_query as _tool_kg_query,
    _tool_kg_schema as _tool_kg_schema,
    _tool_text_search as _tool_text_search,
    agent_query,
    agent_query_standard,
)
from src.retrieval.fusion import QueryResult


COMPLEXITY_THRESHOLD = _complexity_mod.COMPLEXITY_THRESHOLD

_parse_agent_response = parse_agent_response


def _detect_complexity(question: str) -> int:
    return _complexity_mod.detect_complexity(question)


def detect_complexity(question: str) -> int:
    return _detect_complexity(question)


def _decompose_question(question: str) -> list[str] | None:
    return _decomposition_mod.decompose_question(question)


def decompose_question(question: str) -> list[str] | None:
    return _decompose_question(question)


def _synthesize_answers(original_question: str, sub_qa_pairs: list[tuple[str, str]]) -> str:
    return _decomposition_mod.synthesize_answers(original_question, sub_qa_pairs)


def synthesize_answers(original_question: str, sub_qa_pairs: list[tuple[str, str]]) -> str:
    return _synthesize_answers(original_question, sub_qa_pairs)


def _agent_query_with_decomposition(question: str, sub_questions: list[str]) -> QueryResult:
    return _decomposition_mod.agent_query_with_decomposition(question, sub_questions)


def agent_query_with_decomposition(question: str, sub_questions: list[str]) -> QueryResult:
    return _agent_query_with_decomposition(question, sub_questions)


def _agent_query_standard(question: str, top_k: int = 4, emit=None) -> QueryResult:
    return agent_query_standard(question, top_k=top_k, emit=emit)


def _normalize_answer(text: str) -> str:
    return _voting_mod._normalize_answer(text)


def _answers_similar(a: str, b: str) -> bool:
    return _voting_mod._answers_similar(a, b)


def _majority_vote(results: list[QueryResult]) -> QueryResult:
    return _voting_mod._majority_vote(results)


def _run_trajectory(question: str, trajectory_id: int, temperature: float) -> QueryResult:
    return _voting_mod._run_trajectory(question, trajectory_id, temperature)


def run_agent_scaled(question: str, n_trajectories: int | None = None) -> QueryResult:
    """Compatibility wrapper that preserves monkeypatchable module-level helpers."""
    n = n_trajectories if n_trajectories is not None else settings.agent_n_trajectories

    if n <= 1:
        return agent_query(question)

    temperature = settings.agent_temperature_scaled
    results: list[QueryResult] = []

    with ThreadPoolExecutor(max_workers=min(n, 4)) as executor:
        futures = {
            executor.submit(_run_trajectory, question, tid, temperature): tid
            for tid in range(n)
        }
        for future in as_completed(futures):
            results.append(future.result())

    winner = _majority_vote(results)
    winner.retrieval_tier = f"scaled_{n}"
    return winner
