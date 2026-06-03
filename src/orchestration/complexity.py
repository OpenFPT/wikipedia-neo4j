"""Question complexity detection for routing simple vs complex questions."""

from __future__ import annotations

from src.logging_utils import get_logger
from src.prompts import COMPLEXITY_DETECTION_PROMPT

from src.orchestration._parsing import parse_agent_response

logger = get_logger(__name__)

COMPLEXITY_THRESHOLD = 3  # Trigger decomposition for complexity >= 3


def detect_complexity(question: str) -> int:
    """Detect question complexity (1-5) using LLM."""
    try:
        from src.infrastructure.local_llm import chat

        prompt = COMPLEXITY_DETECTION_PROMPT.format(question=question)
        response = chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=100,
            temperature=0.1,
        )

        parsed = parse_agent_response(response)
        if parsed and "complexity" in parsed:
            complexity = int(parsed["complexity"])
            logger.info("Question complexity detected", extra={"complexity": complexity})
            return min(max(complexity, 1), 5)
    except Exception as e:
        logger.warning(f"Complexity detection failed: {e}")

    return 1  # Default to simple if detection fails
