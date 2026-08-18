"""Evaluation pipeline for GraphRAG system using ViWiki-MHR and ViQuAD2.0 datasets."""

from __future__ import annotations

import json
import re
import string
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.logging_utils import get_logger
from src.retrieval.reranker import rerank
from src.retrieval.hybrid import _run_fallback_query, _run_generated_query

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
    except Exception:
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

# --- Ablation Study ---

ABLATION_MODES = ("full_hybrid", "graph_only", "text_only", "no_reranking", "no_multi_hop")

_TEXT_ONLY_CYPHER = """
CALL db.index.fulltext.queryNodes('chunk_text_ft', $q) YIELD node, score
MATCH (p:Page)-[:HAS_CHUNK]->(node)
RETURN p.title AS page_title, p.url AS page_url, p.id AS page_id,
       node.id AS chunk_id, node.text AS chunk_text, score
ORDER BY score DESC
LIMIT $top_k
"""

_GRAPH_ONLY_CYPHER = """
CALL db.index.fulltext.queryNodes('entity_alias_ft', $q) YIELD node, score
MATCH (c:Chunk)-[:MENTIONS]->(node)
MATCH (p:Page)-[:HAS_CHUNK]->(c)
RETURN p.title AS page_title, p.url AS page_url, p.id AS page_id,
       c.id AS chunk_id, c.text AS chunk_text, score
ORDER BY score DESC
LIMIT $top_k
"""


def _retrieve_for_ablation(question: str, mode: str, top_k: int) -> list[dict]:
    """Retrieve chunks using the specified ablation mode.

    Modes:
        full_hybrid — normal hybrid fallback query
        graph_only — only entity-based retrieval via MENTIONS edges
        text_only — only fulltext search on chunk text
        no_reranking — full hybrid (reranking skipped at caller level)
        no_multi_hop — full hybrid (multi-hop skipped at caller level)
    """
    from src.infrastructure.neo4j_client import neo4j_client

    if mode == "text_only":
        with neo4j_client.session() as session:
            records = session.run(_TEXT_ONLY_CYPHER, q=question, top_k=top_k)
            return [dict(r) for r in records]
    elif mode == "graph_only":
        with neo4j_client.session() as session:
            records = session.run(_GRAPH_ONLY_CYPHER, q=question, top_k=top_k)
            return [dict(r) for r in records]
    else:
        # full_hybrid, no_reranking, no_multi_hop all use the same base query
        return _run_fallback_query(question, top_k)


def evaluate_ablation(
    mode: str,
    limit: int = 50,
    top_k_retrieve: int = 20,
    top_k_rerank: int = 5,
    dataset_path: Path | None = None,
) -> EvalMetrics:
    """Run evaluation with a specific ablation mode.

    Args:
        mode: One of ABLATION_MODES controlling which components are active.
        limit: Number of test samples to evaluate.
        top_k_retrieve: Number of chunks to retrieve.
        top_k_rerank: Number of chunks after reranking.
        dataset_path: Optional path to dataset file.

    Returns:
        EvalMetrics with hit rate, MRR, and latency.
    """
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode: {mode!r}. Must be one of {ABLATION_MODES}")

    samples = load_test_set(dataset_path, limit=limit)
    metrics = EvalMetrics(total=len(samples))

    hit_rates: list[float] = []
    mrrs: list[float] = []
    latencies: list[float] = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        metadata = sample.get("metadata", {})
        gold_chunk_ids = metadata.get("evidence_chunk_ids", [])

        if not gold_chunk_ids:
            continue

        t0 = time.perf_counter()
        rows = _retrieve_for_ablation(question, mode, top_k=top_k_retrieve)

        # Apply reranking unless mode disables it
        if mode not in ("no_reranking", "text_only", "graph_only"):
            rows = rerank(question, rows, text_key="chunk_text", top_k=top_k_rerank)

        # Apply multi-hop expansion unless mode disables it
        if mode not in ("no_multi_hop", "text_only", "graph_only", "no_reranking") and rows:
            from src.retrieval.hybrid import _expand_via_links

            page_ids: list[str] = [
                r["page_id"] for r in rows if r.get("page_id")
            ]
            page_ids = list(set(page_ids))
            expanded = _expand_via_links(page_ids, question, top_k_retrieve)
            if expanded:
                seen_chunks = {r["chunk_id"] for r in rows}
                new_rows = [r for r in expanded if r["chunk_id"] not in seen_chunks]
                if new_rows:
                    combined = rows + new_rows
                    rows = rerank(question, combined, text_key="chunk_text", top_k=top_k_rerank)

        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        retrieved_ids = [r.get("chunk_id", "") for r in rows]

        hr = _compute_hit_rate(retrieved_ids, gold_chunk_ids)
        rr = _compute_mrr(retrieved_ids, gold_chunk_ids)
        hit_rates.append(hr)
        mrrs.append(rr)

        metrics.details.append({
            "id": sample.get("id", i),
            "question": question[:80],
            "hit": hr,
            "mrr": rr,
            "latency_ms": round(latency, 1),
        })

        if (i + 1) % 10 == 0:
            logger.info(f"Ablation [{mode}]: {i + 1}/{len(samples)}")

    if hit_rates:
        metrics.context_hit_rate = sum(hit_rates) / len(hit_rates)
        metrics.mrr = sum(mrrs) / len(mrrs)
    if latencies:
        metrics.avg_latency_ms = sum(latencies) / len(latencies)

    return metrics


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

        # Context hit: token overlap between gold context and retrieved chunks
        ctx_tokens = set(context.lower().split())
        hit = 0.0
        if ctx_tokens:
            for rt in retrieved_texts:
                if not rt:
                    continue
                chunk_tokens = set(rt.lower().split())
                overlap = len(ctx_tokens & chunk_tokens) / len(ctx_tokens)
                if overlap >= 0.5:
                    hit = 1.0
                    break
        hit_rates.append(hit)

        # MRR based on token overlap
        rr = 0.0
        if ctx_tokens:
            for rank, rt in enumerate(retrieved_texts, 1):
                if not rt:
                    continue
                chunk_tokens = set(rt.lower().split())
                overlap = len(ctx_tokens & chunk_tokens) / len(ctx_tokens)
                if overlap >= 0.5:
                    rr = 1.0 / rank
                    break
        mrrs.append(rr)

        # Rerank
        reranked = rerank(question, rows, text_key="chunk_text", top_k=top_k_rerank)

        # Generate answer from top reranked chunks
        system_abstains = len(rows) == 0
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


# --- MHQA Dataset Evaluation ---

MHQA_DATASET_PATH = Path("data/mhqa/final.jsonl")


@dataclass
class MHQAMetrics:
    """Aggregated evaluation metrics for MHQA dataset."""

    total: int = 0
    context_hit_rate: float = 0.0
    mrr: float = 0.0
    token_f1: float = 0.0
    supporting_fact_recall: float = 0.0
    decomposition_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    level_breakdown: dict = field(default_factory=dict)
    details: list[dict] = field(default_factory=list)


def _compute_supporting_fact_recall(
    retrieved_chunks: list[dict], supporting_facts: list[dict], context: list[dict]
) -> float:
    """Compute % of gold supporting_facts sentences present in retrieved chunks.

    For each supporting fact (title + sent_id), check if the corresponding
    sentence text appears in any retrieved chunk.
    """
    if not supporting_facts:
        return 0.0

    # Build map of supporting fact sentences
    sf_sentences = []
    for sf in supporting_facts:
        title = sf.get("title", "")
        sent_id = sf.get("sent_id", 0)
        for ctx in context:
            if ctx.get("title") == title:
                sentences = ctx.get("sentences", [])
                if sent_id < len(sentences):
                    sf_sentences.append(sentences[sent_id].lower().strip())
                break

    if not sf_sentences:
        return 0.0

    # Check how many supporting fact sentences appear in retrieved chunks
    retrieved_text = " ".join(c.get("chunk_text", "").lower() for c in retrieved_chunks)
    hits = sum(1 for sent in sf_sentences if sent[:50] in retrieved_text)

    return hits / len(sf_sentences)


def _compute_decomposition_accuracy(
    decomposition: list[dict], top_k: int = 10
) -> float:
    """Run sub-questions independently and check if sub-answers are retrievable."""
    if not decomposition:
        return 0.0

    from src.retrieval.fusion import _wrrf_fuse
    from src.retrieval.bm25 import run_bm25_query
    from src.retrieval.vector import vector_search
    from src.retrieval.graph import graph_search

    hits = 0
    for decomp in decomposition:
        sub_q = decomp.get("sub_question", "")
        sub_a = decomp.get("sub_answer", "").lower().strip()
        if not sub_q or not sub_a:
            continue

        try:
            bm25 = run_bm25_query(sub_q, top_k=top_k)
            vector = vector_search(sub_q, top_k=top_k)
            graph = graph_search(sub_q, top_k=top_k)

            fused = _wrrf_fuse({"bm25": bm25, "vector": vector, "graph": graph})
            for chunk in fused[:top_k]:
                if sub_a in chunk.get("chunk_text", "").lower():
                    hits += 1
                    break
        except Exception:
            continue

    return hits / len(decomposition)


def evaluate_mhqa(
    limit: int | None = None,
    top_k_retrieve: int = 20,
    dataset_path: Path | None = None,
    output_path: str = "reports/eval_mhqa.json",
) -> MHQAMetrics:
    """Evaluate on MHQA dataset with decomposition-aware metrics.

    Metrics:
        - Context Hit Rate: % of questions where answer is in top-K chunks
        - MRR: Mean Reciprocal Rank of first relevant chunk
        - Token F1: Token-level F1 between retrieved answer and gold answer
        - Supporting Fact Recall: % of gold supporting fact sentences in retrieved chunks
        - Decomposition Accuracy: % of sub-questions with retrievable sub-answers
    """
    try:
        from src.retrieval.fusion import _wrrf_fuse
        from src.retrieval.bm25 import run_bm25_query
        from src.retrieval.vector import vector_search
        from src.retrieval.graph import graph_search
        from src.retrieval.community import community_search
    except ImportError as e:
        raise ImportError(f"Retrieval modules not available for MHQA evaluation: {e}") from e

    p = dataset_path or MHQA_DATASET_PATH
    if not p.exists():
        raise FileNotFoundError(f"MHQA dataset not found: {p}. Run the MHQA pipeline first.")

    # Load dataset
    samples = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
                if limit and len(samples) >= limit:
                    break

    logger.info(f"Evaluating MHQA dataset: {len(samples)} samples")
    metrics = MHQAMetrics(total=len(samples))

    hit_rates: list[float] = []
    mrrs: list[float] = []
    f1s: list[float] = []
    sf_recalls: list[float] = []
    decomp_accs: list[float] = []
    latencies: list[float] = []
    level_hits: dict[str, list[float]] = {}

    for i, sample in enumerate(samples):
        question = sample["question"]
        answer = sample["answer"]
        level = sample.get("level", "medium")
        supporting_facts = sample.get("supporting_facts", [])
        context = sample.get("context", [])
        decomposition = sample.get("decomposition", [])

        # Retrieve
        t0 = time.perf_counter()
        try:
            bm25_results = run_bm25_query(question, top_k=top_k_retrieve)
            vector_results = vector_search(question, top_k=top_k_retrieve)
            graph_results = graph_search(question, top_k=top_k_retrieve)
            community_results = community_search(question, top_k=top_k_retrieve)

            fused = _wrrf_fuse(
                {
                    "bm25": bm25_results,
                    "vector": vector_results,
                    "graph": graph_results,
                    "community": community_results,
                },
            )
        except Exception as e:
            logger.warning(f"Retrieval failed for sample {i}: {e}")
            fused = []

        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        # Hit rate: answer in retrieved chunks
        retrieved_texts = [c.get("chunk_text", "").lower() for c in fused[:top_k_retrieve]]
        answer_lower = answer.lower().strip()
        hit = 0.0
        rank = 0
        for j, text in enumerate(retrieved_texts):
            if answer_lower in text:
                hit = 1.0
                rank = j + 1
                break
        hit_rates.append(hit)
        mrrs.append(1.0 / rank if rank > 0 else 0.0)

        # Token F1
        if fused:
            top_chunk_text = fused[0].get("chunk_text", "")
            f1 = compute_token_f1(top_chunk_text, [answer])
        else:
            f1 = 0.0
        f1s.append(f1)

        # Supporting fact recall
        sf_recall = _compute_supporting_fact_recall(fused[:top_k_retrieve], supporting_facts, context)
        sf_recalls.append(sf_recall)

        # Decomposition accuracy (expensive, sample every 5th)
        if i % 5 == 0 and decomposition:
            decomp_acc = _compute_decomposition_accuracy(decomposition, top_k=10)
            decomp_accs.append(decomp_acc)

        # Track by level
        if level not in level_hits:
            level_hits[level] = []
        level_hits[level].append(hit)

        metrics.details.append({
            "id": sample.get("_id", i),
            "question": question[:80],
            "level": level,
            "hit": hit,
            "mrr": 1.0 / rank if rank > 0 else 0.0,
            "sf_recall": round(sf_recall, 3),
            "latency_ms": round(latency, 1),
        })

        if (i + 1) % 20 == 0:
            logger.info(f"MHQA eval: {i + 1}/{len(samples)}")

    # Aggregate
    if hit_rates:
        metrics.context_hit_rate = sum(hit_rates) / len(hit_rates)
        metrics.mrr = sum(mrrs) / len(mrrs)
    if f1s:
        metrics.token_f1 = sum(f1s) / len(f1s)
    if sf_recalls:
        metrics.supporting_fact_recall = sum(sf_recalls) / len(sf_recalls)
    if decomp_accs:
        metrics.decomposition_accuracy = sum(decomp_accs) / len(decomp_accs)
    if latencies:
        metrics.avg_latency_ms = sum(latencies) / len(latencies)

    for level, hits in level_hits.items():
        metrics.level_breakdown[level] = {
            "count": len(hits),
            "hit_rate": sum(hits) / len(hits) if hits else 0.0,
        }

    # Save results
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "total": metrics.total,
        "context_hit_rate": metrics.context_hit_rate,
        "mrr": metrics.mrr,
        "token_f1": metrics.token_f1,
        "supporting_fact_recall": metrics.supporting_fact_recall,
        "decomposition_accuracy": metrics.decomposition_accuracy,
        "avg_latency_ms": metrics.avg_latency_ms,
        "level_breakdown": metrics.level_breakdown,
        "details": metrics.details,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"MHQA results saved to {out}")

    return metrics


def print_mhqa_report(metrics: MHQAMetrics) -> str:
    """Format MHQA evaluation results as a readable report."""
    level_lines = ""
    for level, info in metrics.level_breakdown.items():
        level_lines += f"  {level}: {info['count']} samples, hit_rate={info['hit_rate']:.3f}\n"

    report = f"""
=== MHQA Evaluation Report ===
Total samples: {metrics.total}

--- Retrieval ---
  Context Hit Rate:       {metrics.context_hit_rate:.3f}
  MRR:                    {metrics.mrr:.3f}
  Token F1:               {metrics.token_f1:.3f}

--- Multi-hop Specific ---
  Supporting Fact Recall: {metrics.supporting_fact_recall:.3f}
  Decomposition Accuracy: {metrics.decomposition_accuracy:.3f}

--- By Difficulty ---
{level_lines}
--- Performance ---
  Avg Latency:            {metrics.avg_latency_ms:.0f} ms
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
    elif dataset == "mhqa":
        print(f"Running MHQA evaluation on {limit} samples...")
        mhqa_results = evaluate_mhqa(limit=limit)
        report = print_mhqa_report(mhqa_results)
        print(report)
    else:
        print(f"Running evaluation on {limit} samples...")
        results = evaluate(limit=limit)
        report = print_report(results)
        print(report)
        save_results(results)
