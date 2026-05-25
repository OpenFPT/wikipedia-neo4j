#!/usr/bin/env python3
"""Ablation studies script to measure component contributions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import settings
from src.evaluation import EvalMetrics, evaluate, print_report, save_results
from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment."""

    name: str
    description: str
    multi_hop_expansion: bool
    use_reranking: bool
    use_graph: bool  # If False, use only fulltext search


@dataclass
class AblationResult:
    """Result of an ablation experiment."""

    config: AblationConfig
    metrics: EvalMetrics


# Define ablation configurations
ABLATION_CONFIGS = [
    AblationConfig(
        name="full_hybrid",
        description="Full system: graph + fulltext + reranking + multi-hop",
        multi_hop_expansion=True,
        use_reranking=True,
        use_graph=True,
    ),
    AblationConfig(
        name="no_reranking",
        description="Hybrid without reranking",
        multi_hop_expansion=True,
        use_reranking=False,
        use_graph=True,
    ),
    AblationConfig(
        name="no_multihop",
        description="Hybrid without multi-hop expansion",
        multi_hop_expansion=False,
        use_reranking=True,
        use_graph=True,
    ),
    AblationConfig(
        name="graph_only",
        description="Graph-only (no reranking, no multi-hop)",
        multi_hop_expansion=False,
        use_reranking=False,
        use_graph=True,
    ),
    AblationConfig(
        name="text_only",
        description="Fulltext search only (no graph)",
        multi_hop_expansion=False,
        use_reranking=False,
        use_graph=False,
    ),
]


def _run_ablation_query(
    question: str,
    top_k: int,
    config: AblationConfig,
) -> list[dict]:
    """Run retrieval with ablation configuration."""
    from src.retrieve import _expand_via_links, _run_fallback_query

    if not config.use_graph:
        # Text-only: just fulltext search
        rows = _run_fallback_query(question, top_k)
        return rows

    # Graph-based retrieval
    rows = _run_fallback_query(question, top_k)

    if config.use_reranking and rows:
        from src.reranker import rerank

        rows = rerank(question, rows, text_key="chunk_text", top_k=top_k)

    if config.multi_hop_expansion and rows:
        page_ids = list({r.get("page_id") for r in rows if r.get("page_id")})
        expanded = _expand_via_links(page_ids, question, top_k)
        if expanded:
            seen_chunks = {r["chunk_id"] for r in rows}
            new_rows = [r for r in expanded if r["chunk_id"] not in seen_chunks]
            if new_rows:
                combined = rows + new_rows
                if config.use_reranking:
                    from src.reranker import rerank

                    rows = rerank(question, combined, text_key="chunk_text", top_k=top_k)
                else:
                    rows = combined

    return rows


def run_ablation_study(
    limit: int = 20,
    top_k: int = 20,
    dataset_path: Path | None = None,
    output_dir: str = "reports",
) -> list[AblationResult]:
    """Run ablation study with all configurations.

    Args:
        limit: Number of samples to evaluate per configuration
        top_k: Number of chunks to retrieve
        dataset_path: Path to dataset file
        output_dir: Directory to save results

    Returns:
        List of ablation results
    """
    from src.evaluation import load_test_set

    logger.info(f"Loading test set (limit={limit})...")
    samples = load_test_set(dataset_path, limit=limit)

    if not samples:
        logger.error("No samples loaded")
        return []

    logger.info(f"Loaded {len(samples)} samples")

    results = []

    for config in ABLATION_CONFIGS:
        logger.info(f"Running ablation: {config.name} - {config.description}")

        hit_rates = []
        mrrs = []
        latencies = []

        import time

        for i, sample in enumerate(samples):
            question = sample["question"]
            metadata = sample.get("metadata", {})
            gold_chunk_ids = metadata.get("evidence_chunk_ids", [])

            if not gold_chunk_ids:
                continue

            t0 = time.perf_counter()
            rows = _run_ablation_query(question, top_k, config)
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)

            retrieved_ids = [r.get("chunk_id", "") for r in rows]

            # Compute hit rate and MRR
            hit = 1.0 if any(gid in retrieved_ids for gid in gold_chunk_ids) else 0.0
            hit_rates.append(hit)

            rr = 0.0
            for rank, rid in enumerate(retrieved_ids, 1):
                if rid in gold_chunk_ids:
                    rr = 1.0 / rank
                    break
            mrrs.append(rr)

            if (i + 1) % 5 == 0:
                logger.info(f"  {config.name}: {i + 1}/{len(samples)}")

        # Aggregate metrics
        metrics = EvalMetrics(total=len(samples))
        if hit_rates:
            metrics.context_hit_rate = sum(hit_rates) / len(hit_rates)
            metrics.mrr = sum(mrrs) / len(mrrs)
        if latencies:
            metrics.avg_latency_ms = sum(latencies) / len(latencies)

        results.append(AblationResult(config=config, metrics=metrics))

        logger.info(
            f"  {config.name}: HR={metrics.context_hit_rate:.3f}, "
            f"MRR={metrics.mrr:.3f}, Latency={metrics.avg_latency_ms:.0f}ms"
        )

    # Save results
    output_path = Path(output_dir) / "ablation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total_samples": len(samples),
        "results": [
            {
                "config": {
                    "name": r.config.name,
                    "description": r.config.description,
                    "multi_hop_expansion": r.config.multi_hop_expansion,
                    "use_reranking": r.config.use_reranking,
                    "use_graph": r.config.use_graph,
                },
                "metrics": asdict(r.metrics),
            }
            for r in results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Ablation results saved to {output_path}")

    return results


def print_ablation_report(results: list[AblationResult]) -> str:
    """Format ablation study results as a readable report."""
    report = "=== Ablation Study Results ===\n\n"
    report += "| Configuration | Description | Hit Rate | MRR | Latency (ms) |\n"
    report += "|---|---|---|---|---|\n"

    for result in results:
        report += (
            f"| {result.config.name} | {result.config.description} | "
            f"{result.metrics.context_hit_rate:.3f} | {result.metrics.mrr:.3f} | "
            f"{result.metrics.avg_latency_ms:.0f} |\n"
        )

    if results:
        baseline = results[0]
        report += "\n### Component Contributions\n\n"

        for result in results[1:]:
            hr_delta = (result.metrics.context_hit_rate - baseline.metrics.context_hit_rate) * 100
            mrr_delta = (result.metrics.mrr - baseline.metrics.mrr) * 100
            latency_delta = result.metrics.avg_latency_ms - baseline.metrics.avg_latency_ms

            report += f"**{result.config.name}** vs baseline:\n"
            report += f"- Hit Rate: {hr_delta:+.1f}%\n"
            report += f"- MRR: {mrr_delta:+.1f}%\n"
            report += f"- Latency: {latency_delta:+.0f}ms\n\n"

    return report


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ablation studies on GraphRAG system")
    parser.add_argument("--limit", type=int, default=20, help="Number of samples per configuration")
    parser.add_argument("--top-k", type=int, default=20, help="Number of chunks to retrieve")
    parser.add_argument(
        "--dataset", type=str, default=None, help="Path to dataset file (default: data/viwiki_mhr.jsonl)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports", help="Directory to save results"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else None
    results = run_ablation_study(
        limit=args.limit,
        top_k=args.top_k,
        dataset_path=dataset_path,
        output_dir=args.output_dir,
    )

    report = print_ablation_report(results)
    print(report)

    # Save report as markdown
    report_path = Path(args.output_dir) / "ablation_results.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
