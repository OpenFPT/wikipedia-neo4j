"""Backward-compatibility shim: imports redirect to src.orchestration submodules."""

from src.config import settings  # noqa: F401
from src.infrastructure.neo4j_client import neo4j_client  # noqa: F401

from src.orchestration._parsing import parse_agent_response  # noqa: F401
from src.orchestration._parsing import parse_agent_response as _parse_agent_response  # noqa: F401

from src.orchestration.agent_loop import (  # noqa: F401
    agent_query,
    agent_query_standard,
    _tool_entity_neighborhood,
    _tool_path_search,
    _tool_kg_schema,
    _tool_kg_query,
    _tool_text_search,
    _tool_get_passage,
)
from src.orchestration.voting import (  # noqa: F401
    run_agent_scaled,
    _majority_vote,
    _answers_similar,
)
from src.retrieval.fusion import QueryResult  # noqa: F401
