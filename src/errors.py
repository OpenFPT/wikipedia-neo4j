"""Shared API error schema and helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Stable error codes returned by the API."""

    cypher_generation_failed = "cypher_generation_failed"
    ingest_failed = "ingest_failed"
    invalid_request = "invalid_request"
    key_config_invalid = "key_config_invalid"


class ErrorResponse(BaseModel):
    """Standard error payload returned for all API failures."""

    error_code: str
    message: str
    request_id: str
    hint: str | None = None


def error_payload(
    error_code: ErrorCode,
    message: str,
    request_id: str,
    hint: str | None = None,
) -> ErrorResponse:
    """Build a response-ready error payload."""
    payload = ErrorResponse(
        error_code=error_code.value,
        message=message,
        request_id=request_id,
        hint=hint,
    )
    return payload


def normalize_error_detail(detail: Any, status_code: int) -> tuple[ErrorCode, str, str | None]:
    """Normalize FastAPI HTTPException detail into standard fields."""
    if isinstance(detail, dict):
        code_value = detail.get("error_code") or detail.get("code")
        message = detail.get("message") or detail.get("detail") or "Request failed"
        hint = detail.get("hint")
        if code_value in {c.value for c in ErrorCode}:
            return ErrorCode(code_value), str(message), hint
        return ErrorCode.invalid_request, str(message), hint

    if status_code >= 500:
        return ErrorCode.cypher_generation_failed, str(detail), None
    return ErrorCode.invalid_request, str(detail), None
