from __future__ import annotations

import pytest

import src.infrastructure.llm as llm


def test_strip_code_fence() -> None:
    s = "```json\n{\"cypher\":\"MATCH\"}\n```"
    assert llm._strip_code_fence(s) == '{"cypher":"MATCH"}'


def test_is_retryable_gemini_error() -> None:
    assert llm._is_retryable_gemini_error(RuntimeError("429 quota exceeded")) is True
    assert llm._is_retryable_gemini_error(RuntimeError("syntax error")) is False


def test_assert_readonly_cypher_blocks_call_dbms() -> None:
    cypher = (
        "CALL dbms.procedures() "
        "RETURN '' AS page_title, '' AS page_url, '' AS chunk_id, '' AS chunk_text, 0 AS score"
    )
    with pytest.raises(RuntimeError, match="not read-only"):
        llm.assert_readonly_cypher(cypher)
