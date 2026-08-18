"""Evaluate the GraphRAG system on the MHQA dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.evaluation import evaluate_mhqa, print_mhqa_report
from src.logging_utils import configure_logging

configure_logging(settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="eval_mhqa")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG on MHQA dataset")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--dataset", type=str, default="data/mhqa/final.jsonl", help="MHQA dataset path")
    parser.add_argument("--output", type=str, default="reports/eval_mhqa.json", help="Output report path")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K for retrieval")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Run the MHQA pipeline first:")
        print("  uv run python scripts/mhqa_extract_walks.py")
        print("  uv run python scripts/mhqa_generate.py")
        print("  uv run python scripts/mhqa_qc.py")
        print("  uv run python scripts/mhqa_retrievability.py")
        sys.exit(1)

    print(f"Evaluating MHQA dataset: {dataset_path}")
    if args.limit:
        print(f"  Limit: {args.limit} samples")

    metrics = evaluate_mhqa(
        limit=args.limit,
        top_k_retrieve=args.top_k,
        dataset_path=dataset_path,
        output_path=args.output,
    )

    report = print_mhqa_report(metrics)
    print(report)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
