"""Cross-encoder reranking for retrieval results."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

_reranker: CrossEncoder | None = None
_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _get_reranker() -> CrossEncoder | None:
    global _reranker
    if _reranker is None:
        try:
            logger.info("Loading cross-encoder reranker", extra={"model": _RERANKER_MODEL})
            _reranker = CrossEncoder(_RERANKER_MODEL, max_length=512, device="cpu")
        except Exception as e:
            logger.error("Failed to load reranker model", extra={"error": str(e)})
            return None
    return _reranker


def rerank(
    query: str,
    documents: list[dict],
    text_key: str = "chunk_text",
    top_k: int = 5,
    min_score: float | None = None,
) -> list[dict]:
    """Rerank retrieved documents using cross-encoder.

    Args:
        query: The user question.
        documents: List of dicts, each must contain `text_key`.
        text_key: Key in each dict holding the passage text.
        top_k: Number of top results to return after reranking.
        min_score: Minimum cross-encoder score to keep a passage.
                   Defaults to settings.rerank_min_score.

    Returns:
        Top-k documents sorted by cross-encoder relevance score,
        filtered to those above min_score.
    """
    if not documents:
        return []

    if min_score is None:
        min_score = settings.rerank_min_score

    model = _get_reranker()
    if model is None:
        logger.warning("Reranker unavailable, returning documents unranked")
        return documents[:top_k]

    pairs: list[tuple[str, str]] = [(query, (doc.get(text_key) or "")[:512]) for doc in documents]
    scores = model.predict(pairs)  # type: ignore[arg-type]

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
    filtered = [d for d in ranked if d["rerank_score"] >= min_score]

    if not filtered and ranked:
        filtered = ranked[:1]

    logger.info(
        "Reranked results",
        extra={"input": len(documents), "above_threshold": len(filtered), "output": min(top_k, len(filtered))},
    )
    return filtered[:top_k]
