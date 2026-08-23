"""Ollama chat wrapper (local LLM via HTTP)."""

from __future__ import annotations

from typing import Any

import requests

from src.config import settings
from src.logging_utils import get_logger


logger = get_logger(__name__)


def chat(
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Call Ollama's /api/chat.

    messages format: [{"role":"system"|"user"|"assistant","content":"..."}]
    """
    url = settings.ollama_base_url.rstrip("/") + "/api/chat"
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_new_tokens),
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        msg = (data.get("message") or {}).get("content")
        return (msg or "").strip()
    except Exception as exc:
        logger.warning("Ollama chat failed", extra={"error": str(exc), "url": url, "model": settings.ollama_model})
        raise
