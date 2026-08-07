"""Dispatch chat calls to the configured backend (local HF vs Ollama)."""

from __future__ import annotations

from src.config import settings


def chat(messages: list[dict[str, str]], max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    if settings.model_mode == "ollama":
        from src.infrastructure.ollama_llm import chat as _chat

        return _chat(messages, max_new_tokens=max_new_tokens, temperature=temperature)

    # Default: in-process HF model (may require torch/bnb working)
    from src.infrastructure.local_llm import chat as _chat

    return _chat(messages, max_new_tokens=max_new_tokens, temperature=temperature)

