"""Evaluation pipeline for GraphRAG system using ViWiki-MHR dataset."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.logging_utils import get_logger
from src.reranker import rerank
from src.retrieve import _run_fallback_query, _run_generated_query

logger = get_logger(__name__)

DATASET_PATH = Path("data/viwiki_mhr.jsonl")


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""

    total: int = 0
    context_hit_rate: float = 0.0
    mrr: float = 0.0
    avg_latency_ms: float = 0.0
    rerank_context_hit_rate: float = 0.0
    rerank_mrr: float = 0.0
    details: list[dict] = field(default_factory=list)


def load_test_set(path: Path | None = None, limit: int | None = None) -> list[dict]:
    """Load ViWiki-MHR test questions."""
    p = path or DATASET_PATH
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    samples = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def _retrieve_chunks(question: str, top_k: int = 20) -> list[dict]:
    """Retrieve chunks using fallback (fulltext) for evaluation."""
    try:
        rows = _run_generated_query(question, top_k)
    except (RuntimeError, ValueError, KeyError, TypeError):
        rows = _run_fallback_query(question, top_k)
    return rows


def _compute_hit_rate(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Check if any gold passage appears in retrieved results."""
    if not gold_ids:
        return 0.0
    return 1.0 if any(gid in retrieved_ids for gid in gold_ids) else 0.0


def _compute_mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    if not gold_ids:
        return 0.0
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in gold_ids:
            return 1.0 / i
    return 0.0


def evaluate(
    limit: int = 100,
    top_k_retrieve: int = 20,
    top_k_rerank: int = 5,
    dataset_path: Path | None = None,
) -> EvalMetrics:
    """Run evaluation on ViWiki-MHR dataset.

    Measures retrieval quality before and after reranking.
    """
    samples = load_test_set(dataset_path, limit=limit)
    metrics = EvalMetrics(total=len(samples))

    hit_rates = []
    mrrs = []
    rerank_hit_rates = []
    rerank_mrrs = []
    latencies = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        metadata = sample.get("metadata", {})
        gold_chunk_ids = metadata.get("evidence_chunk_ids", [])

        if not gold_chunk_ids:
            continue

        t0 = time.perf_counter()
        rows = _retrieve_chunks(question, top_k=top_k_retrieve)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        retrieved_ids = [r.get("chunk_id", "") for r in rows]

        hr = _compute_hit_rate(retrieved_ids, gold_chunk_ids)
        rr = _compute_mrr(retrieved_ids, gold_chunk_ids)
        hit_rates.append(hr)
        mrrs.append(rr)

        reranked = rerank(question, rows, text_key="chunk_text", top_k=top_k_rerank)
        reranked_ids = [r.get("chunk_id", "") for r in reranked]

        rhr = _compute_hit_rate(reranked_ids, gold_chunk_ids)
        rmrr = _compute_mrr(reranked_ids, gold_chunk_ids)
        rerank_hit_rates.append(rhr)
        rerank_mrrs.append(rmrr)

        metrics.details.append({
            "id": sample.get("id", i),
            "question": question[:80],
            "hit": hr,
            "mrr": rr,
            "rerank_hit": rhr,
            "rerank_mrr": rmrr,
            "latency_ms": round(latency, 1),
        })

        if (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i + 1}/{len(samples)}")

    if hit_rates:
        metrics.context_hit_rate = sum(hit_rates) / len(hit_rates)
        metrics.mrr = sum(mrrs) / len(mrrs)
        metrics.rerank_context_hit_rate = sum(rerank_hit_rates) / len(rerank_hit_rates)
        metrics.rerank_mrr = sum(rerank_mrrs) / len(rerank_mrrs)
    if latencies:
        metrics.avg_latency_ms = sum(latencies) / len(latencies)

    return metrics


def print_report(metrics: EvalMetrics) -> str:
    """Format evaluation results as a readable report."""
    report = f"""
=== GraphRAG Evaluation Report ===
Total samples: {metrics.total}

--- Retrieval (before reranking) ---
  Context Hit Rate: {metrics.context_hit_rate:.3f}
  MRR:             {metrics.mrr:.3f}

--- Retrieval (after reranking) ---
  Context Hit Rate: {metrics.rerank_context_hit_rate:.3f}
  MRR:             {metrics.rerank_mrr:.3f}

--- Performance ---
  Avg Latency:     {metrics.avg_latency_ms:.0f} ms

--- Improvement from Reranking ---
  Hit Rate: {metrics.context_hit_rate:.3f} -> {metrics.rerank_context_hit_rate:.3f} ({(metrics.rerank_context_hit_rate - metrics.context_hit_rate) * 100:+.1f}%)
  MRR:      {metrics.mrr:.3f} -> {metrics.rerank_mrr:.3f} ({(metrics.rerank_mrr - metrics.mrr) * 100:+.1f}%)
"""
    return report


def save_results(metrics: EvalMetrics, output_path: str = "reports/eval_results.json") -> None:
    """Save evaluation results to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "total": metrics.total,
        "context_hit_rate": metrics.context_hit_rate,
        "mrr": metrics.mrr,
        "rerank_context_hit_rate": metrics.rerank_context_hit_rate,
        "rerank_mrr": metrics.rerank_mrr,
        "avg_latency_ms": metrics.avg_latency_ms,
        "details": metrics.details,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {out}")


if __name__ == "__main__":
    import sys

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"Running evaluation on {limit} samples...")
    results = evaluate(limit=limit)
    report = print_report(results)
    print(report)
    save_results(results)
