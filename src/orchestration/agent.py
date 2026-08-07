"""ReAct agent loop with graph tools for multi-hop question answering.

This module re-exports from the split submodules for backward compatibility.
"""

# Re-export public interface from submodules
from src.orchestration.agent_loop import (  # noqa: F401
    agent_query,
    agent_query_standard,
    SYSTEM_PROMPT,
    TOOLS,
    MAX_ITERATIONS,
)
from src.orchestration.voting import run_agent_scaled  # noqa: F401
from src.orchestration.complexity import detect_complexity, COMPLEXITY_THRESHOLD  # noqa: F401
from src.orchestration.decomposition import (  # noqa: F401
    decompose_question,
    synthesize_answers,
    agent_query_with_decomposition,
)
from src.orchestration._parsing import parse_agent_response  # noqa: F401
