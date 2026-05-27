"""Evaluate retrieval quality directly (no LLM needed).

Calls hybrid_retrieve() — the same function backing the MCP search_knowledge_base tool —
and measures context hit rate + MRR against ViQuAD2 gold answers.

Usage:
    uv run python scripts/eval_retrieval_direct.py --limit 100
    uv run python scripts/eval_retrieval_direct.py --dataset data/viquad2/validation.jsonl --top-k 10
    uv run python scripts/eval_retrieval_direct.py --limit 500 --output reports/eval_retrieval_500.json
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieve import hybrid_retrieve


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    return text.strip()


def answer_in_context(gold_answers: list[str], chunks: list[dict]) -> tuple[bool, int | None]:
    """Check if any gold answer appears in retrieved chunks. Returns (hit, rank)."""
    for gold in gold_answers:
        norm_gold = normalize(gold)
        if not norm_gold:
            continue
        gold_tokens = set(norm_gold.split())
        for rank, chunk in enumerate(chunks, 1):
            chunk_text = normalize(chunk.get("chunk_text", ""))
            chunk_tokens = set(chunk_text.split())
            overlap = gold_tokens & chunk_tokens
            if len(overlap) / len(gold_tokens) >= 0.6:
                return True, rank
    return False, None


def load_dataset(path: Path, limit: int | None = None) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            gold = item.get("gold_answers") or []
            if not gold and "answers" in item:
                gold = item["answers"].get("text", [])
            if not gold:
                continue
            samples.append({"id": item.get("id", ""), "question": item["question"], "gold_answers": gold})
            if limit and len(samples) >= limit:
                break
    return samples


def main():
    parser = argparse.ArgumentParser(description="Direct retrieval evaluation on ViQuAD2")
    parser.add_argument("--dataset", default="data/viquad2/validation.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    samples = load_dataset(dataset_path, args.limit)
    print(f"Loaded {len(samples)} questions from {dataset_path}")

    hits = 0
    mrr_sum = 0.0
    total = 0
    errors = 0
    details = []
    latencies = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        gold_answers = sample["gold_answers"]

        t0 = time.time()
        try:
            results = hybrid_retrieve(question, top_k=args.top_k)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR [{i+1}]: {e}")
            details.append({"id": sample["id"], "question": question, "hit": False, "error": str(e)})
            continue
        elapsed = time.time() - t0
        latencies.append(elapsed)

        hit, rank = answer_in_context(gold_answers, results)
        total += 1
        if hit:
            hits += 1
            mrr_sum += 1.0 / rank

        details.append({
            "id": sample["id"],
            "question": question,
            "gold_answers": gold_answers,
            "hit": hit,
            "rank": rank,
            "n_results": len(results),
            "latency_ms": round(elapsed * 1000, 1),
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            hr = hits / total if total else 0
            mrr = mrr_sum / total if total else 0
            avg_lat = sum(latencies) / len(latencies) * 1000 if latencies else 0
            print(f"  [{i+1}/{len(samples)}] hit_rate={hr:.3f} MRR={mrr:.3f} avg_latency={avg_lat:.0f}ms errors={errors}")

    # Final metrics
    hit_rate = hits / total if total else 0
    mrr = mrr_sum / total if total else 0
    avg_latency = sum(latencies) / len(latencies) * 1000 if latencies else 0

    report = {
        "dataset": str(dataset_path),
        "total_evaluated": total,
        "top_k": args.top_k,
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "errors": errors,
        "details": details,
    }

    print(f"\n{'='*50}")
    print(f"Results: {total} questions evaluated")
    print(f"  Context Hit Rate: {hit_rate:.1%}")
    print(f"  MRR:              {mrr:.4f}")
    print(f"  Avg Latency:      {avg_latency:.0f}ms")
    print(f"  Errors:           {errors}")
    print(f"{'='*50}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
