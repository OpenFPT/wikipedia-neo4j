"""Inference-time scaling: parallel trajectory sampling + majority voting."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import settings
from src.logging_utils import get_logger
from src.retrieval.fusion import QueryResult

from src.orchestration._parsing import parse_agent_response
from src.orchestration.agent_loop import (
    SYSTEM_PROMPT,
    TOOLS,
    MAX_ITERATIONS,
    _extract_citations,
    _synthesize_from_observations,
    agent_query,
)

logger = get_logger(__name__)

_DIVERSITY_NUDGES = [
    "",  # trajectory 0: no nudge (baseline)
    "Try a different search strategy than usual. Start with text_search instead of kg_query.",
    "Focus on entity_neighborhood and path_search tools to explore the graph structure.",
    "Use multiple short Cypher queries rather than one complex query. Explore step by step.",
    "Start by searching for related entities first, then look for specific evidence.",
    "Try to find the answer through page links and entity co-occurrence patterns.",
]


def _normalize_answer(text: str) -> str:
    """Normalize an answer for comparison: strip, lowercase, remove trailing punctuation."""
    normalized = text.strip().lower()
    normalized = re.sub(r"[.\s]+$", "", normalized)
    return normalized


def _answers_similar(a: str, b: str) -> bool:
    """Check if two answers are similar enough to be grouped together.

    Uses exact match on normalized form, or containment check for
    cases where one answer is a more detailed version of another.
    """
    norm_a = _normalize_answer(a)
    norm_b = _normalize_answer(b)

    if norm_a == norm_b:
        return True

    # One contains the other (handles cases like "Hà Nội" vs "Hà Nội, Việt Nam")
    if len(norm_a) > 10 and len(norm_b) > 10:
        shorter = norm_a if len(norm_a) <= len(norm_b) else norm_b
        longer = norm_b if len(norm_a) <= len(norm_b) else norm_a
        if shorter in longer:
            return True

    return False


def _majority_vote(results: list[QueryResult]) -> QueryResult:
    """Select the best answer from multiple trajectory results via majority voting.

    Grouping strategy:
    1. Exact match on normalized answers
    2. Containment check (one answer is substring of another)

    Tie-breaking: pick the answer with the most citations.
    """
    if not results:
        return QueryResult(
            answer="Không tìm thấy thông tin phù hợp trong cơ sở tri thức.",
            citations=[],
        )

    if len(results) == 1:
        return results[0]

    # Group similar answers
    groups: list[list[QueryResult]] = []

    for result in results:
        placed = False
        for group in groups:
            if _answers_similar(result.answer, group[0].answer):
                group.append(result)
                placed = True
                break
        if not placed:
            groups.append([result])

    # Sort groups: largest group first, then by max citations in group
    groups.sort(
        key=lambda g: (len(g), max(len(r.citations) for r in g)),
        reverse=True,
    )

    winning_group = groups[0]

    # Within the winning group, pick the result with the most citations
    winner = max(winning_group, key=lambda r: len(r.citations))

    logger.info(
        "Majority vote completed",
        extra={
            "n_trajectories": len(results),
            "n_groups": len(groups),
            "winning_group_size": len(winning_group),
            "winner_citations": len(winner.citations),
        },
    )

    return winner


def _run_trajectory(question: str, trajectory_id: int, temperature: float) -> QueryResult:
    """Run a single agent trajectory with diversity nudge.

    Each trajectory uses a slightly modified system prompt to encourage
    exploration of different graph paths.
    """
    from src.infrastructure.local_llm import chat

    # Build system prompt with optional diversity nudge
    nudge = _DIVERSITY_NUDGES[trajectory_id % len(_DIVERSITY_NUDGES)]
    if nudge:
        system_prompt = SYSTEM_PROMPT + f"\n\nNote: {nudge}"
    else:
        system_prompt = SYSTEM_PROMPT

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    citations: list[dict] = []
    observations: list[str] = []
    actions_taken: list[str] = []
    seen_chunk_ids: set[str] = set()

    for iteration in range(MAX_ITERATIONS):
        llm_attempts = 0
        raw = None
        while llm_attempts < 2:
            try:
                raw = chat(messages, max_new_tokens=512, temperature=temperature)
                break
            except Exception as e:
                llm_attempts += 1
                if llm_attempts >= 2:
                    logger.warning(
                        "Trajectory LLM call failed after retry",
                        extra={"trajectory_id": trajectory_id, "iteration": iteration, "error": str(e)},
                    )
                else:
                    logger.debug(
                        "Trajectory LLM call failed, retrying",
                        extra={"trajectory_id": trajectory_id, "iteration": iteration, "error": str(e)},
                    )

        if raw is None:
            break

        parsed = parse_agent_response(raw)
        if not parsed:
            logger.warning(
                "Trajectory produced unparseable output",
                extra={"trajectory_id": trajectory_id, "iteration": iteration, "raw": raw[:200]},
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Please respond with valid JSON."})
            continue

        if "final_answer" in parsed:
            if not actions_taken:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "You must use at least one tool before providing a final answer. "
                    "Use kg_query or text_search to find evidence first.",
                })
                continue
            answer = parsed["final_answer"]
            logger.info(
                "Trajectory converged",
                extra={"trajectory_id": trajectory_id, "iterations": iteration + 1},
            )
            return QueryResult(answer=answer, citations=citations)

        action = parsed.get("action", "")
        action_input = parsed.get("action_input", {})
        if isinstance(action_input, str):
            action_input = {"query": action_input} if action == "text_search" else {"cypher": action_input}

        tool_fn = TOOLS.get(action)
        if not tool_fn:
            observation = (
                f"Error: Unknown tool '{action}'. "
                "Available: kg_schema, kg_query, text_search, get_passage, entity_neighborhood, path_search"
            )
        else:
            observation = tool_fn(action_input)

        observations.append(observation)
        actions_taken.append(action)

        _extract_citations(observation, seen_chunk_ids, citations)

        messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
        messages.append({"role": "user", "content": f"Observation: {observation[:2000]}"})

        logger.debug(
            "Trajectory iteration",
            extra={"trajectory_id": trajectory_id, "iteration": iteration, "action": action},
        )

    logger.warning(
        "Trajectory did not converge",
        extra={"trajectory_id": trajectory_id, "iterations": MAX_ITERATIONS},
    )
    return _synthesize_from_observations(observations, citations)


def run_agent_scaled(question: str, n_trajectories: int | None = None) -> QueryResult:
    """Run inference-time scaled agent with parallel trajectory sampling and majority voting.

    Inspired by Inference-Scaled GraphRAG: multiple independent reasoning trajectories
    explore different graph paths, then majority voting selects the most consistent answer.

    Args:
        question: The question to answer.
        n_trajectories: Number of parallel trajectories to run. If None, uses
            settings.agent_n_trajectories. If 1, falls back to standard agent_query.

    Returns:
        QueryResult with the majority-voted answer and retrieval_tier="scaled_{n}".
    """
    n = n_trajectories if n_trajectories is not None else settings.agent_n_trajectories

    if n <= 1:
        return agent_query(question)

    temperature = settings.agent_temperature_scaled

    logger.info(
        "Starting scaled agent inference",
        extra={"n_trajectories": n, "temperature": temperature},
    )

    results: list[QueryResult] = []

    # Run trajectories in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(n, 4)) as executor:
        futures = {
            executor.submit(_run_trajectory, question, tid, temperature): tid
            for tid in range(n)
        }

        for future in as_completed(futures):
            tid = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(
                    "Trajectory failed with exception",
                    extra={"trajectory_id": tid, "error": str(e)},
                )

    if not results:
        logger.error("All trajectories failed")
        return QueryResult(
            answer="Không tìm thấy thông tin phù hợp trong cơ sở tri thức.",
            citations=[],
        )

    # Majority vote to select the best answer
    winner = _majority_vote(results)

    # Tag the retrieval tier to indicate scaled inference was used
    return QueryResult(
        answer=winner.answer,
        citations=winner.citations,
        retrieval_tier=f"scaled_{n}",
    )
