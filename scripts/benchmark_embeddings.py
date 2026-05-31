"""Benchmark embedding models for Vietnamese retrieval quality.

Compares models on MRR@10, Hit@5, NDCG@10, and inference latency
using query-passage pairs from evaluation data.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MODELS = [
    settings.local_embedding_model,
    "AITeamVN/Vietnamese_Embedding_v2",
]

DEFAULT_EVAL_PATH = Path("data/eval/viquad2_sample.jsonl")
FALLBACK_EVAL_PATH = Path("data/viwiki_mhr.jsonl")


def load_eval_data(path: Path, limit: int | None = None) -> list[dict]:
    """Load evaluation samples with 'question' and 'context'/'passages' fields."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation data not found: {path}")

    samples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Normalize: ensure we have question and at least one gold passage
            question = item.get("question") or item.get("query", "")
            if not question:
                continue

            # Support multiple field names for gold passages
            passages: list[str] = []
            if "context" in item:
                ctx = item["context"]
                if isinstance(ctx, list):
                    passages = [str(c) for c in ctx]
                else:
                    passages = [str(ctx)]
            elif "passages" in item:
                passages = [str(p) for p in item["passages"]]
            elif "gold_passages" in item:
                passages = [str(p) for p in item["gold_passages"]]
            elif "answer_passage" in item:
                passages = [str(item["answer_passage"])]
            elif "answer" in item and item["answer"]:
                passages = [str(item["answer"])]

            if not passages:
                continue

            samples.append({"question": question, "gold_passages": passages})
            if limit and len(samples) >= limit:
                break

    return samples


def cosine_similarity_matrix(queries: np.ndarray, passages: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and passage embeddings."""
    # Normalize
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
    p_norm = passages / (np.linalg.norm(passages, axis=1, keepdims=True) + 1e-10)
    return q_norm @ p_norm.T


def compute_mrr(rankings: list[list[int]], k: int = 10) -> float:
    """Compute Mean Reciprocal Rank at k."""
    mrr_sum = 0.0
    for relevant_positions in rankings:
        for rank, pos in enumerate(relevant_positions[:k], start=1):
            if pos == 1:
                mrr_sum += 1.0 / rank
                break
    return mrr_sum / len(rankings) if rankings else 0.0


def compute_hit_rate(rankings: list[list[int]], k: int = 5) -> float:
    """Compute Hit Rate at k (fraction of queries with at least one hit in top-k)."""
    hits = 0
    for relevant_positions in rankings:
        if any(pos == 1 for pos in relevant_positions[:k]):
            hits += 1
    return hits / len(rankings) if rankings else 0.0


def compute_ndcg(rankings: list[list[int]], k: int = 10) -> float:
    """Compute NDCG at k."""
    ndcg_sum = 0.0
    for relevant_positions in rankings:
        dcg = 0.0
        for rank, pos in enumerate(relevant_positions[:k], start=1):
            if pos == 1:
                dcg += 1.0 / math.log2(rank + 1)
        # Ideal DCG: all relevant docs at the top
        num_relevant = sum(relevant_positions)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(num_relevant, k)))
        ndcg_sum += dcg / idcg if idcg > 0 else 0.0
    return ndcg_sum / len(rankings) if rankings else 0.0


def benchmark_model(
    model_id: str,
    samples: list[dict],
) -> dict:
    """Benchmark a single embedding model on the evaluation samples.

    Returns dict with metrics: mrr@10, hit@5, ndcg@10, avg_query_time_ms,
    avg_passage_time_ms, embedding_dim.
    """
    logger.info(f"Loading model: {model_id}")
    load_start = time.perf_counter()
    model = SentenceTransformer(model_id, trust_remote_code=True)
    load_time = time.perf_counter() - load_start
    logger.info(f"Model loaded in {load_time:.1f}s, dim={model.get_sentence_embedding_dimension()}")

    # Collect all unique passages across samples for a shared corpus
    all_passages: list[str] = []
    passage_to_idx: dict[str, int] = {}
    sample_gold_indices: list[list[int]] = []

    for sample in samples:
        gold_idxs: list[int] = []
        for passage in sample["gold_passages"]:
            if passage not in passage_to_idx:
                passage_to_idx[passage] = len(all_passages)
                all_passages.append(passage)
            gold_idxs.append(passage_to_idx[passage])
        sample_gold_indices.append(gold_idxs)

    queries = [s["question"] for s in samples]

    # Embed queries
    logger.info(f"Embedding {len(queries)} queries...")
    q_start = time.perf_counter()
    query_embeddings = model.encode(queries, show_progress_bar=False, batch_size=32)
    q_time = time.perf_counter() - q_start
    avg_query_time_ms = (q_time / len(queries)) * 1000

    # Embed passages
    logger.info(f"Embedding {len(all_passages)} passages...")
    p_start = time.perf_counter()
    passage_embeddings = model.encode(all_passages, show_progress_bar=False, batch_size=32)
    p_time = time.perf_counter() - p_start
    avg_passage_time_ms = (p_time / len(all_passages)) * 1000

    # Compute similarity and rank
    query_emb = np.array(query_embeddings)
    passage_emb = np.array(passage_embeddings)
    sim_matrix = cosine_similarity_matrix(query_emb, passage_emb)

    # For each query, rank all passages by similarity and mark relevance
    rankings: list[list[int]] = []
    for i, gold_idxs in enumerate(sample_gold_indices):
        gold_set = set(gold_idxs)
        sorted_indices = np.argsort(-sim_matrix[i])
        relevance = [1 if int(idx) in gold_set else 0 for idx in sorted_indices]
        rankings.append(relevance)

    mrr_10 = compute_mrr(rankings, k=10)
    hit_5 = compute_hit_rate(rankings, k=5)
    ndcg_10 = compute_ndcg(rankings, k=10)

    results = {
        "model_id": model_id,
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "num_queries": len(queries),
        "num_passages": len(all_passages),
        "mrr@10": mrr_10,
        "hit@5": hit_5,
        "ndcg@10": ndcg_10,
        "avg_query_time_ms": avg_query_time_ms,
        "avg_passage_time_ms": avg_passage_time_ms,
        "model_load_time_s": load_time,
    }

    logger.info(
        f"Results for {model_id}: MRR@10={mrr_10:.4f}, Hit@5={hit_5:.4f}, "
        f"NDCG@10={ndcg_10:.4f}, query={avg_query_time_ms:.2f}ms/q"
    )

    # Free GPU memory
    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return results


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table of benchmark results."""
    print("\n" + "=" * 90)
    print("EMBEDDING MODEL BENCHMARK RESULTS")
    print("=" * 90)

    # Header
    header = (
        f"{'Model':<45} {'Dim':>4} {'MRR@10':>7} {'Hit@5':>6} "
        f"{'NDCG@10':>8} {'ms/query':>9}"
    )
    print(header)
    print("-" * 90)

    # Rows
    for r in results:
        # Shorten model name for display
        name = r["model_id"]
        if len(name) > 44:
            name = "..." + name[-41:]
        row = (
            f"{name:<45} {r['embedding_dim']:>4} {r['mrr@10']:>7.4f} "
            f"{r['hit@5']:>6.4f} {r['ndcg@10']:>8.4f} {r['avg_query_time_ms']:>9.2f}"
        )
        print(row)

    print("-" * 90)
    print(f"{'Queries':<45} {results[0]['num_queries']:>4}")
    print(f"{'Corpus passages':<45} {results[0]['num_passages']:>4}")
    print("=" * 90)

    # Winner summary
    best_mrr = max(results, key=lambda x: x["mrr@10"])
    best_speed = min(results, key=lambda x: x["avg_query_time_ms"])
    print(f"\nBest MRR@10:  {best_mrr['model_id']} ({best_mrr['mrr@10']:.4f})")
    print(f"Fastest:      {best_speed['model_id']} ({best_speed['avg_query_time_ms']:.2f} ms/query)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models for Vietnamese retrieval"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model IDs to benchmark (default: GreenNode + AITeamVN)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of test queries to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to evaluation JSONL (default: data/eval/viquad2_sample.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON",
    )
    args = parser.parse_args()

    # Resolve dataset path
    if args.dataset:
        eval_path = Path(args.dataset)
    elif DEFAULT_EVAL_PATH.exists():
        eval_path = DEFAULT_EVAL_PATH
    elif FALLBACK_EVAL_PATH.exists():
        eval_path = FALLBACK_EVAL_PATH
    else:
        raise FileNotFoundError(
            f"No evaluation data found. Provide --dataset or place data at "
            f"{DEFAULT_EVAL_PATH} or {FALLBACK_EVAL_PATH}"
        )

    logger.info(f"Loading evaluation data from {eval_path}")
    samples = load_eval_data(eval_path, limit=args.limit)
    if not samples:
        raise ValueError(f"No valid samples loaded from {eval_path}")
    logger.info(f"Loaded {len(samples)} evaluation samples")

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_ids:
        raise ValueError("No models specified")

    logger.info(f"Benchmarking {len(model_ids)} models: {model_ids}")

    all_results: list[dict] = []
    for model_id in model_ids:
        try:
            result = benchmark_model(model_id, samples)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to benchmark {model_id}: {e}")
            all_results.append({
                "model_id": model_id,
                "error": str(e),
                "embedding_dim": 0,
                "num_queries": len(samples),
                "num_passages": 0,
                "mrr@10": 0.0,
                "hit@5": 0.0,
                "ndcg@10": 0.0,
                "avg_query_time_ms": 0.0,
                "avg_passage_time_ms": 0.0,
                "model_load_time_s": 0.0,
            })

    # Print comparison
    print_comparison_table(all_results)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
