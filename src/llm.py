"""Gemini integration for embeddings and Cypher generation."""

from __future__ import annotations

import json
import random
import re
import time

from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

from src.config import (
    load_gemini_api_keys,
    resolve_cypher_model,
    settings,
)
from src.logging_utils import get_logger


logger = get_logger(__name__)

_local_embedding_model: SentenceTransformer | None = None


def _is_retryable_gemini_error(exc: Exception) -> bool:
    """Return whether an exception message indicates retryable API failure."""
    msg = str(exc).lower()
    retry_tokens = [
        "429",
        "rate",
        "quota",
        "unauthorized",
        "forbidden",
        "failed_precondition",
        "location is not supported",
        "api key",
    ]
    return any(tok in msg for tok in retry_tokens)


def _client_pool() -> list[genai.Client]:
    """Create a client per configured Gemini API key."""
    keys = load_gemini_api_keys()
    return [genai.Client(api_key=key) for key in keys]


def _get_local_embedding_model() -> SentenceTransformer:
    global _local_embedding_model
    if _local_embedding_model is None:
        _local_embedding_model = SentenceTransformer(settings.local_embedding_model)
    return _local_embedding_model


def _embed_texts_local(texts: list[str]) -> list[list[float]]:
    model = _get_local_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [vec.tolist() for vec in vectors]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for texts with key-rotation fallback."""
    if settings.embedding_backend == "local":
        return _embed_texts_local(texts)
    clients = _client_pool()
    last_error: Exception | None = None

    for i, client in enumerate(clients, start=1):
        try:
            vectors: list[list[float]] = []
            for text in texts:
                resp = client.models.embed_content(
                    model=settings.gemini_model_embedding,
                    contents=text,
                )
                emb_list = getattr(resp, "embeddings", None)
                if not emb_list:
                    raise RuntimeError("Gemini embedding response had no vectors")
                vectors.append(list(emb_list[0].values))
            logger.debug("Embedding generation succeeded", extra={"client_index": i, "count": len(texts)})
            return vectors
        except Exception as exc:
            last_error = exc
            logger.warning("Embedding generation failed", extra={"client_index": i, "error": str(exc)})
            if not _is_retryable_gemini_error(exc):
                raise
            delay = min(2**i, 16) + random.uniform(0, 1)
            time.sleep(delay)
            continue

    # All Gemini keys exhausted — raise so callers can skip or handle gracefully.
    raise RuntimeError(f"All Gemini keys failed for embedding generation: {last_error}")


def _strip_code_fence(s: str) -> str:
    """Strip markdown code fences from model output."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


_CYPHER_SCHEMA = (
    "Nodes: Page(id,title,url,summary), Chunk(id,text,sequence_number,embedding), "
    "Entity(id,name,type) with optional labels Person/Organization/Location/Work. "
    "Relationships: (Page)-[:HAS_CHUNK]->(Chunk), (Chunk)-[:MENTIONS]->(Entity)."
)

_CYPHER_SYSTEM_PROMPT = """You are a Neo4j Cypher generator.
Strict rules:
- Read-only only. No CREATE, MERGE, DELETE, SET, DROP, CALL dbms/procedures writes.
- Return fields exactly as: page_title, page_url, chunk_id, chunk_text, score.
- Use MATCH/WHERE with safe logic.
- LIMIT 8 max.
- Return ONLY JSON object: {"cypher":"..."}"""


def _build_cypher_user_prompt(question: str) -> str:
    return f"Generate ONE read-only Cypher query for this question: {question}\n\nSchema: {_CYPHER_SCHEMA}"


def _generate_cypher_local(question: str) -> str:
    """Generate Cypher using the local SLM."""
    from src.local_llm import chat

    messages = [
        {"role": "system", "content": _CYPHER_SYSTEM_PROMPT},
        {"role": "user", "content": _build_cypher_user_prompt(question)},
    ]
    raw = chat(messages, max_new_tokens=512, temperature=0.1)
    text = _strip_code_fence(raw)
    parsed = json.loads(text)
    cypher = str(parsed.get("cypher", "")).strip()
    if not cypher:
        raise RuntimeError("Local model returned empty Cypher")
    if "$top_k" not in cypher and "limit" not in cypher.lower():
        cypher = f"{cypher.rstrip(';')} LIMIT $top_k"
    logger.debug("Cypher generation succeeded (local model)")
    return cypher


def _generate_cypher_gemini(question: str) -> str:
    """Generate Cypher using Gemini API with key rotation."""
    prompt = f"{_CYPHER_SYSTEM_PROMPT}\n\n{_build_cypher_user_prompt(question)}"

    clients = _client_pool()
    last_error: Exception | None = None

    for i, client in enumerate(clients, start=1):
        try:
            resp = client.models.generate_content(
                model=resolve_cypher_model(),
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                ),
            )
            text = _strip_code_fence(resp.text or "")
            parsed = json.loads(text)
            cypher = str(parsed.get("cypher", "")).strip()
            if not cypher:
                raise RuntimeError("Gemini returned empty Cypher")
            if "$top_k" not in cypher and "limit" not in cypher.lower():
                cypher = f"{cypher.rstrip(';')} LIMIT $top_k"
            logger.debug("Cypher generation succeeded", extra={"client_index": i})
            return cypher
        except Exception as exc:
            last_error = exc
            logger.warning("Cypher generation failed", extra={"client_index": i, "error": str(exc)})
            if not _is_retryable_gemini_error(exc):
                raise
            delay = min(2**i, 16) + random.uniform(0, 1)
            time.sleep(delay)
            continue

    raise RuntimeError(f"All Gemini keys failed for cypher generation: {last_error}")


def generate_readonly_cypher(question: str) -> str:
    """Generate a read-only Cypher query for a natural-language question."""
    if settings.model_mode == "local":
        return _generate_cypher_local(question)
    return _generate_cypher_gemini(question)


def assert_readonly_cypher(cypher: str) -> None:
    """Validate that generated Cypher is read-only and shape-compatible."""
    raw = (cypher or "").strip()
    if not raw:
        raise RuntimeError("Generated Cypher is empty")

    trimmed = raw[:-1] if raw.endswith(";") else raw
    if ";" in trimmed:
        raise RuntimeError("Generated Cypher contains multiple statements")

    lowered = re.sub(r"\s+", " ", raw.lower())
    blocked = [
        " create ",
        " merge ",
        " delete ",
        " detach ",
        " set ",
        " remove ",
        " drop ",
        " load csv",
        " apoc.periodic",
        " call dbms",
    ]
    padded = f" {lowered} "
    if any(token in padded for token in blocked):
        raise RuntimeError("Generated Cypher is not read-only")

    required_aliases = ["page_title", "page_url", "chunk_id", "chunk_text", "score"]
    for alias in required_aliases:
        if re.search(rf"\bas\s+{re.escape(alias)}\b", lowered) is None:
            raise RuntimeError(f"Generated Cypher missing required alias: {alias}")
