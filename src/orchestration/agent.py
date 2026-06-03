"""ReAct agent loop with graph tools for multi-hop question answering.

This module re-exports from the split submodules for backward compatibility.
"""

from src.infrastructure.neo4j_client import neo4j_client  # noqa: F401
from src.orchestration._parsing import parse_agent_response  # noqa: F401
from src.orchestration.agent_loop import (  # noqa: F401
    MAX_ITERATIONS,
    SYSTEM_PROMPT,
    TOOLS,
    _check_sufficiency,
    _tool_get_passage,
    _tool_kg_query,
    _tool_text_search,
    agent_query,
    agent_query_standard,
)
from src.orchestration.complexity import COMPLEXITY_THRESHOLD, detect_complexity  # noqa: F401
from src.orchestration.decomposition import (  # noqa: F401
    agent_query_with_decomposition,
    decompose_question,
    synthesize_answers,
)
from src.orchestration.voting import run_agent_scaled  # noqa: F401

_parse_agent_response = parse_agent_response
