"""Evaluate graph-structural QA via kg_query (Approach A).

Tests whether Cypher queries return correct answers from the knowledge graph.

Usage:
    uv run python scripts/eval_graph_qa.py
    uv run python scripts/eval_graph_qa.py --output reports/eval_graph_qa.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infrastructure.neo4j_client import neo4j_client
from src.mcp_pkg.tools import _validate_readonly_cypher


def normalize_answer(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower().strip())
    text = re.sub(r"[()（）]", "", text)
    return text


def extract_numeric(text: str) -> str | None:
    m = re.search(r"[\d,.]+", text)
    return m.group(0).replace(",", "") if m else None


def check_answer(expected: str, actual_results: list[dict]) -> tuple[bool, str]:
    if not actual_results:
        return False, "no results"

    actual_str = json.dumps(actual_results, ensure_ascii=False)

    expected_num = extract_numeric(expected)
    if expected_num:
        for row in actual_results:
            for v in row.values():
                if str(v) == expected_num:
                    return True, str(v)
                if isinstance(v, float) and expected_num.replace(".", "") in str(v).replace(".", ""):
                    return True, str(v)

    norm_expected = normalize_answer(expected)
    norm_actual = normalize_answer(actual_str)
    if norm_expected in norm_actual:
        return True, actual_str[:200]

    expected_parts = [p.strip() for p in re.split(r"[,，]", expected) if p.strip()]
    if len(expected_parts) > 1:
        matches = sum(1 for p in expected_parts if normalize_answer(p) in norm_actual)
        if matches >= len(expected_parts) * 0.5:
            return True, f"{matches}/{len(expected_parts)} parts matched"

    return False, actual_str[:200]


def run_cypher(cypher: str) -> list[dict]:
    _validate_readonly_cypher(cypher)
    with neo4j_client.session() as session:
        result = session.run(cypher)
        return [dict(r) for r in result]


def main():
    parser = argparse.ArgumentParser(description="Evaluate graph-structural QA via kg_query")
    parser.add_argument("--dataset", default="data/eval/graph_structural_qa.jsonl")
    parser.add_argument("--output", default="reports/eval_graph_qa.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    samples = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} graph QA samples")

    correct = 0
    errors = 0
    details = []

    for i, sample in enumerate(samples):
        t0 = time.time()
        try:
            results = run_cypher(sample["cypher"])
            elapsed = time.time() - t0
            hit, actual = check_answer(sample["answer"], results)
            if hit:
                correct += 1
            details.append({
                "question": sample["question"],
                "type": sample["type"],
                "expected": sample["answer"],
                "actual": actual,
                "correct": hit,
                "latency_ms": round(elapsed * 1000, 1),
            })
        except Exception as e:
            errors += 1
            details.append({
                "question": sample["question"],
                "type": sample["type"],
                "expected": sample["answer"],
                "error": str(e),
                "correct": False,
            })

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            acc = correct / (i + 1 - errors) if (i + 1 - errors) > 0 else 0
            print(f"  [{i+1}/{len(samples)}] accuracy={acc:.1%} errors={errors}")

    total_evaluated = len(samples) - errors
    accuracy = correct / total_evaluated if total_evaluated > 0 else 0

    by_type = {}
    for d in details:
        t = d["type"]
        if t not in by_type:
            by_type[t] = {"correct": 0, "total": 0}
        if not d.get("error"):
            by_type[t]["total"] += 1
            if d["correct"]:
                by_type[t]["correct"] += 1

    report = {
        "dataset": str(dataset_path),
        "total_samples": len(samples),
        "total_evaluated": total_evaluated,
        "accuracy": round(accuracy, 4),
        "errors": errors,
        "by_type": {k: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0} for k, v in by_type.items()},
        "details": details,
    }

    print(f"\n{'='*50}")
    print(f"Graph QA Eval Results ({total_evaluated} questions)")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Errors:    {errors}")
    for t, v in by_type.items():
        print(f"  {t}: {v['correct']}/{v['total']} ({v['correct']/v['total']:.0%})" if v['total'] > 0 else f"  {t}: 0/0")
    print(f"{'='*50}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
