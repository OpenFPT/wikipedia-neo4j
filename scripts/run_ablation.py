#!/usr/bin/env python3
"""Ablation studies: run the system with components disabled to measure each one's contribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation import ABLATION_MODES, EvalMetrics, evaluate_ablation
from src.logging_utils import get_logger

logger = get_logger(__name__)


def _print_comparison_table(results: dict[str, EvalMetrics]) -> str:
    """Format ablation results as a comparison table."""
    header = (
        f"{'Configuration':<16} {'Hit Rate':>10} {'MRR':>10} {'Avg Latency (ms)':>18}"
    )
    sep = "-" * len(header)
    lines = [
        "",
        "=== Ablation Study Results ===",
        "",
        header,
        sep,
    ]

    for mode, metrics in results.items():
        lines.append(
            f"{mode:<16} {metrics.context_hit_rate:>10.3f} {metrics.mrr:>10.3f} "
            f"{metrics.avg_latency_ms:>18.1f}"
        )

    # Compute deltas vs baseline
    baseline = results.get("full_hybrid")
    if baseline:
        lines.append("")
        lines.append("--- Component Contributions (delta vs full_hybrid) ---")
        lines.append("")
        for mode, metrics in results.items():
            if mode == "full_hybrid":
                continue
            hr_delta = (metrics.context_hit_rate - baseline.context_hit_rate) * 100
            mrr_delta = (metrics.mrr - baseline.mrr) * 100
            lat_delta = metrics.avg_latency_ms - baseline.avg_latency_ms
            lines.append(
                f"  {mode:<14}: HR {hr_delta:+.1f}%  MRR {mrr_delta:+.1f}%  "
                f"Latency {lat_delta:+.0f}ms"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run ablation study across all configurations."""
    parser = argparse.ArgumentParser(
        description="Run ablation studies on the GraphRAG retrieval pipeline"
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=50, help="Number of test samples (default: 50)"
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="Number of chunks to retrieve (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/ablation_results.json",
        help="Output JSON path (default: reports/ablation_results.json)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Path to test dataset JSONL"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else None
    results: dict[str, EvalMetrics] = {}

    print(f"Running ablation study: {len(ABLATION_MODES)} configurations, {args.limit} samples each")
    print()

    for mode in ABLATION_MODES:
        logger.info(f"Starting ablation mode: {mode}")
        print(f"  [{mode}] running...", end="", flush=True)

        metrics = evaluate_ablation(
            mode=mode,
            limit=args.limit,
            top_k_retrieve=args.top_k,
            dataset_path=dataset_path,
        )
        results[mode] = metrics
        print(
            f" done. HR={metrics.context_hit_rate:.3f} MRR={metrics.mrr:.3f} "
            f"Latency={metrics.avg_latency_ms:.0f}ms"
        )

    # Print comparison table
    report = _print_comparison_table(results)
    print(report)

    # Save JSON results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "configurations": {
            mode: {
                "hit_rate": round(m.context_hit_rate, 4),
                "mrr": round(m.mrr, 4),
                "avg_latency_ms": round(m.avg_latency_ms, 1),
            }
            for mode, m in results.items()
        },
        "limit": args.limit,
        "top_k": args.top_k,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
