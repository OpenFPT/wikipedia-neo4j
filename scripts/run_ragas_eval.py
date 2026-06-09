#!/usr/bin/env python3
"""CLI for RAGAS evaluation on ViWiki-MHR dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation import (
    compute_ragas_metrics,
    load_test_set,
    print_ragas_report,
    save_ragas_results,
)
from src.logging_utils import get_logger
from src.retrieval.hybrid import _run_fallback_query, _run_generated_query

logger = get_logger(__name__)


def _retrieve_chunks(question: str, top_k: int = 20) -> list[dict]:
    """Retrieve chunks using fallback (fulltext) for evaluation."""
    try:
        rows = _run_generated_query(question, top_k)
    except (RuntimeError, ValueError, KeyError, TypeError):
        rows = _run_fallback_query(question, top_k)
    return rows


def run_ragas_eval(
    limit: int = 100,
    top_k: int = 20,
    dataset_path: Path | None = None,
    output_path: str = "reports/eval_ragas_results.json",
) -> None:
    """Run RAGAS evaluation on ViWiki-MHR dataset.

    Args:
        limit: Number of samples to evaluate
        top_k: Number of chunks to retrieve per question
        dataset_path: Path to dataset file
        output_path: Path to save results
    """
    logger.info(f"Loading test set (limit={limit})...")
    samples = load_test_set(dataset_path, limit=limit)

    if not samples:
        logger.error("No samples loaded")
        return

    logger.info(f"Loaded {len(samples)} samples")

    questions = []
    contexts_list = []
    answers = []
    ground_truths = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        questions.append(question)

        # Retrieve chunks
        rows = _retrieve_chunks(question, top_k=top_k)
        context_texts = [r.get("chunk_text", "") for r in rows if r.get("chunk_text")]
        contexts_list.append(context_texts)

        # Generate answer from top chunks (simple concatenation for now)
        answer = " ".join(context_texts[:3]) if context_texts else ""
        answers.append(answer)

        # Ground truth (if available)
        metadata = sample.get("metadata", {})
        gold_answer = metadata.get("answer", "")
        ground_truths.append(gold_answer)

        if (i + 1) % 10 == 0:
            logger.info(f"Prepared {i + 1}/{len(samples)}")

    logger.info("Computing RAGAS metrics...")
    metrics = compute_ragas_metrics(
        questions=questions,
        contexts=contexts_list,
        answers=answers,
        ground_truths=ground_truths if any(ground_truths) else None,
    )

    report = print_ragas_report(metrics)
    print(report)

    save_ragas_results(metrics, output_path)
    logger.info(f"Results saved to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on ViWiki-MHR")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--top-k", type=int, default=20, help="Number of chunks to retrieve")
    parser.add_argument(
        "--dataset", type=str, default=None, help="Path to dataset file (default: data/viwiki_mhr.jsonl)"
    )
    parser.add_argument(
        "--output", type=str, default="reports/eval_ragas_results.json", help="Output path for results"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else None
    run_ragas_eval(
        limit=args.limit,
        top_k=args.top_k,
        dataset_path=dataset_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
