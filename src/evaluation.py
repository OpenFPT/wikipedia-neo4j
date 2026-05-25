"""Evaluation pipeline for GraphRAG system using ViWiki-MHR and ViQuAD2.0 datasets."""

from __future__ import annotations

import json
import re
import string
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.logging_utils import get_logger
from src.reranker import rerank
from src.retrieve import _run_fallback_query, _run_generated_query

logger = get_logger(__name__)

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

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


@dataclass
class RAGASMetrics:
    """RAGAS evaluation metrics for answer quality."""

    total: int = 0
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
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


# --- RAGAS Evaluation ---


def compute_ragas_metrics(
    questions: list[str],
    contexts: list[list[str]],
    answers: list[str],
    ground_truths: list[str] | None = None,
) -> RAGASMetrics:
    """Compute RAGAS metrics for QA pairs with retrieved contexts.

    Args:
        questions: List of questions
        contexts: List of context lists (one per question)
        answers: List of generated answers
        ground_truths: Optional list of ground truth answers

    Returns:
        RAGASMetrics with computed scores
    """
    # Validate inputs first
    if not (len(questions) == len(contexts) == len(answers)):
        raise ValueError("questions, contexts, and answers must have same length")

    if ground_truths and len(ground_truths) != len(questions):
        raise ValueError("ground_truths must match length of questions")

    if not RAGAS_AVAILABLE:  # pragma: no cover
        logger.warning("RAGAS not available, skipping RAGAS metrics")
        return RAGASMetrics(total=len(questions))

    try:  # pragma: no cover
        from typing import Any

        from datasets import Dataset

        # Build dataset in RAGAS format
        data: dict[str, Any] = {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        # Compute metrics
        logger.info(f"Computing RAGAS metrics for {len(questions)} samples...")
        result: Any = ragas_evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )

        metrics = RAGASMetrics(total=len(questions))

        # Extract aggregated scores
        if "context_precision" in result:
            metrics.context_precision = float(result["context_precision"].mean())
        if "context_recall" in result:
            metrics.context_recall = float(result["context_recall"].mean())
        if "faithfulness" in result:
            metrics.faithfulness = float(result["faithfulness"].mean())
        if "answer_relevancy" in result:
            metrics.answer_relevancy = float(result["answer_relevancy"].mean())

        # Store per-sample details
        for i in range(len(questions)):
            detail = {
                "question": questions[i][:80],
                "context_precision": (
                    float(result["context_precision"][i])
                    if "context_precision" in result
                    else None
                ),
                "context_recall": (
                    float(result["context_recall"][i])
                    if "context_recall" in result
                    else None
                ),
                "faithfulness": (
                    float(result["faithfulness"][i]) if "faithfulness" in result else None
                ),
                "answer_relevancy": (
                    float(result["answer_relevancy"][i])
                    if "answer_relevancy" in result
                    else None
                ),
            }
            metrics.details.append(detail)

        logger.info(
            f"RAGAS metrics computed: "
            f"context_precision={metrics.context_precision:.3f}, "
            f"context_recall={metrics.context_recall:.3f}, "
            f"faithfulness={metrics.faithfulness:.3f}, "
            f"answer_relevancy={metrics.answer_relevancy:.3f}"
        )
        return metrics

    except Exception as e:  # pragma: no cover
        logger.error(f"Error computing RAGAS metrics: {e}")
        return RAGASMetrics(total=len(questions))


def print_ragas_report(metrics: RAGASMetrics) -> str:
    """Format RAGAS evaluation results as a readable report."""
    report = f"""
=== RAGAS Evaluation Report ===
Total samples: {metrics.total}

--- Context Quality ---
  Context Precision: {metrics.context_precision:.3f}
  Context Recall:    {metrics.context_recall:.3f}

--- Answer Quality ---
  Faithfulness:      {metrics.faithfulness:.3f}
  Answer Relevancy:  {metrics.answer_relevancy:.3f}
"""
    return report


def save_ragas_results(
    metrics: RAGASMetrics, output_path: str = "reports/eval_ragas_results.json"
) -> None:
    """Save RAGAS evaluation results to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "total": metrics.total,
        "context_precision": metrics.context_precision,
        "context_recall": metrics.context_recall,
        "faithfulness": metrics.faithfulness,
        "answer_relevancy": metrics.answer_relevancy,
        "details": metrics.details,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"RAGAS results saved to {out}")

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def _normalize_answer(text: str) -> str:
    """Normalize answer for comparison: lowercase, strip punct, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def compute_em(prediction: str, gold_answers: list[str]) -> float:
    """Exact Match: 1.0 if normalized prediction matches any gold answer."""
    if not gold_answers:
        return 0.0
    norm_pred = _normalize_answer(prediction)
    for gold in gold_answers:
        if _normalize_answer(gold) == norm_pred:
            return 1.0
    return 0.0


def compute_token_f1(prediction: str, gold_answers: list[str]) -> float:
    """Token-level F1: best F1 across all gold answers."""
    if not gold_answers:
        return 0.0
    norm_pred = _normalize_answer(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = sum(1 for t in pred_tokens if t in gold_tokens)
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


@dataclass
class ViQuADMetrics:
    """Aggregated evaluation metrics for ViQuAD2.0."""

    total: int = 0
    answerable_count: int = 0
    impossible_count: int = 0
    context_hit_rate: float = 0.0
    mrr: float = 0.0
    exact_match: float = 0.0
    token_f1: float = 0.0
    abstain_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    details: list[dict] = field(default_factory=list)


def evaluate_viquad(
    limit: int = 100,
    top_k_retrieve: int = 20,
    top_k_rerank: int = 5,
    split: str = "validation",
) -> ViQuADMetrics:
    """Run evaluation on UIT-ViQuAD2.0 dataset."""
    from src.viquad_adapter import load_viquad

    samples = load_viquad(split=split, limit=limit)
    metrics = ViQuADMetrics(total=len(samples))

    hit_rates: list[float] = []
    mrrs: list[float] = []
    ems: list[float] = []
    f1s: list[float] = []
    abstain_correct: list[float] = []
    latencies: list[float] = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        gold_answers = sample["gold_answers"]
        is_impossible = sample["is_impossible"]
        context = sample["context"]

        if is_impossible:
            metrics.impossible_count += 1
        else:
            metrics.answerable_count += 1

        t0 = time.perf_counter()
        rows = _retrieve_chunks(question, top_k=top_k_retrieve)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        retrieved_texts = [r.get("chunk_text", "") for r in rows]
        max_score = max((r.get("score", 0) for r in rows), default=0)

        # Context hit: check if gold context appears in any retrieved chunk
        ctx_norm = context[:200].lower()
        hit = 1.0 if any(ctx_norm in rt.lower() for rt in retrieved_texts if rt) else 0.0
        hit_rates.append(hit)

        # MRR based on context match
        rr = 0.0
        for rank, rt in enumerate(retrieved_texts, 1):
            if rt and ctx_norm in rt.lower():
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        # Rerank
        reranked = rerank(question, rows, text_key="chunk_text", top_k=top_k_rerank)

        # Generate answer from top reranked chunks
        system_abstains = len(rows) == 0 or max_score < 0.1
        if system_abstains:
            predicted_answer = ""
        else:
            predicted_answer = " ".join(
                r.get("chunk_text", "")[:200] for r in reranked if r.get("chunk_text")
            )

        # Metrics
        if is_impossible:
            abstain_correct.append(1.0 if system_abstains else 0.0)
        else:
            if gold_answers:
                ems.append(compute_em(predicted_answer, gold_answers))
                f1s.append(compute_token_f1(predicted_answer, gold_answers))

        metrics.details.append({
            "id": sample["id"],
            "question": question[:80],
            "is_impossible": is_impossible,
            "hit": hit,
            "mrr": rr,
            "em": ems[-1] if ems and not is_impossible else None,
            "f1": f1s[-1] if f1s and not is_impossible else None,
            "latency_ms": round(latency, 1),
        })

        if (i + 1) % 10 == 0:
            logger.info(f"ViQuAD eval: {i + 1}/{len(samples)}")

    if hit_rates:
        metrics.context_hit_rate = sum(hit_rates) / len(hit_rates)
        metrics.mrr = sum(mrrs) / len(mrrs)
    if ems:
        metrics.exact_match = sum(ems) / len(ems)
        metrics.token_f1 = sum(f1s) / len(f1s)
    if abstain_correct:
        metrics.abstain_accuracy = sum(abstain_correct) / len(abstain_correct)
    if latencies:
        metrics.avg_latency_ms = sum(latencies) / len(latencies)

    return metrics


def print_viquad_report(metrics: ViQuADMetrics) -> str:
    """Format ViQuAD2.0 evaluation results."""
    report = f"""
=== ViQuAD2.0 Evaluation Report ===
Total samples: {metrics.total} (answerable: {metrics.answerable_count}, impossible: {metrics.impossible_count})

--- Retrieval ---
  Context Hit Rate: {metrics.context_hit_rate:.3f}
  MRR:             {metrics.mrr:.3f}

--- Answer Quality ---
  Exact Match:     {metrics.exact_match:.3f}
  Token F1:        {metrics.token_f1:.3f}

--- Abstain ---
  Abstain Accuracy: {metrics.abstain_accuracy:.3f}

--- Performance ---
  Avg Latency:     {metrics.avg_latency_ms:.0f} ms
"""
    return report


if __name__ == "__main__":
    import sys

    dataset = "viwiki_mhr"
    limit = 50

    for arg in sys.argv[1:]:
        if arg.startswith("--dataset="):
            dataset = arg.split("=", 1)[1]
        elif arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])
        elif arg.isdigit():
            limit = int(arg)

    if dataset == "viquad2":
        print(f"Running ViQuAD2.0 evaluation on {limit} samples...")
        viquad_results = evaluate_viquad(limit=limit)
        report = print_viquad_report(viquad_results)
        print(report)
        out = Path("reports/eval_viquad2_results.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total": viquad_results.total,
            "answerable_count": viquad_results.answerable_count,
            "impossible_count": viquad_results.impossible_count,
            "context_hit_rate": viquad_results.context_hit_rate,
            "mrr": viquad_results.mrr,
            "exact_match": viquad_results.exact_match,
            "token_f1": viquad_results.token_f1,
            "abstain_accuracy": viquad_results.abstain_accuracy,
            "avg_latency_ms": viquad_results.avg_latency_ms,
            "details": viquad_results.details,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {out}")
    else:
        print(f"Running evaluation on {limit} samples...")
        results = evaluate(limit=limit)
        report = print_report(results)
        print(report)
        save_results(results)
