"""Cross-encoder reranking for retrieval results."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.logging_utils import get_logger

logger = get_logger(__name__)

_reranker: CrossEncoder | None = None
_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder reranker", extra={"model": _RERANKER_MODEL})
        _reranker = CrossEncoder(_RERANKER_MODEL, max_length=512)
    return _reranker


def rerank(query: str, documents: list[dict], text_key: str = "chunk_text", top_k: int = 5) -> list[dict]:
    """Rerank retrieved documents using cross-encoder.

    Args:
        query: The user question.
        documents: List of dicts, each must contain `text_key`.
        text_key: Key in each dict holding the passage text.
        top_k: Number of top results to return after reranking.

    Returns:
        Top-k documents sorted by cross-encoder relevance score.
    """
    if not documents:
        return []

    model = _get_reranker()
    pairs = [[query, (doc.get(text_key) or "")[:512]] for doc in documents]
    scores = model.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
    logger.info("Reranked results", extra={"input": len(documents), "output": top_k})
    return ranked[:top_k]
