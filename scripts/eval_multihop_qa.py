"""Evaluate multi-hop QA via answer_question MCP tool (Approach B).

Tests end-to-end QA accuracy: question decomposition → retrieval → synthesis.
Uses the same agent pipeline as the MCP answer_question tool.

Usage:
    uv run python scripts/eval_multihop_qa.py
    uv run python scripts/eval_multihop_qa.py --limit 10 --output reports/eval_multihop_qa.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestration.agent import run_agent_scaled


MULTIHOP_QUESTIONS = [
    {
        "question": "Trang Paris liên kết đến những quốc gia nào trong đồ thị tri thức?",
        "gold_answers": ["Pháp", "Nhật Bản", "Việt Nam", "Bồ Đào Nha", "Trung Quốc", "Đức", "Thổ Nhĩ Kỳ", "Luxembourg"],
        "type": "listing",
    },
    {
        "question": "Có bao nhiêu tổ chức được nhắc đến trong bài viết về Chiến tranh Đông Dương?",
        "gold_answers": ["63"],
        "type": "aggregation",
    },
    {
        "question": "Tổ chức nào được nhắc đến nhiều nhất trong toàn bộ đồ thị tri thức?",
        "gold_answers": ["Quốc hội Hoa Kỳ"],
        "type": "aggregation",
    },
    {
        "question": "Đảng Cộng sản Đông Dương được nhắc đến trong bài viết Wikipedia nào?",
        "gold_answers": ["Chiến tranh Đông Dương"],
        "type": "listing",
    },
    {
        "question": "Trang nào có nhiều liên kết đi ra nhất trong đồ thị?",
        "gold_answers": ["Paris"],
        "type": "aggregation",
    },
    {
        "question": "Châu Âu và Châu Á, địa điểm nào được nhắc đến nhiều hơn trong đồ thị tri thức?",
        "gold_answers": ["Châu Âu"],
        "type": "comparison",
    },
    {
        "question": "Những trang nào vừa liên kết đến Paris vừa liên kết đến Pháp?",
        "gold_answers": ["Chiến tranh Đông Dương", "Người Do Thái", "Mông Cổ", "Roma", "François Mitterrand"],
        "type": "path",
    },
    {
        "question": "Trang Donald Trump có bao nhiêu đoạn văn (chunk)?",
        "gold_answers": ["396"],
        "type": "aggregation",
    },
    {
        "question": "Liên minh châu Âu và Đảng Cộng sản Việt Nam, tổ chức nào được nhắc đến nhiều hơn?",
        "gold_answers": ["Liên minh châu Âu"],
        "type": "comparison",
    },
    {
        "question": "Có bao nhiêu cộng đồng (community) trong đồ thị tri thức?",
        "gold_answers": ["7"],
        "type": "aggregation",
    },
    {
        "question": "Cộng đồng lớn nhất có bao nhiêu thành viên?",
        "gold_answers": ["467"],
        "type": "aggregation",
    },
    {
        "question": "Trang California đề cập đến bao nhiêu tổ chức khác nhau?",
        "gold_answers": ["73"],
        "type": "aggregation",
    },
    {
        "question": "Trang nào đề cập nhiều địa điểm nhất?",
        "gold_answers": ["California"],
        "type": "comparison",
    },
    {
        "question": "Trang nào đề cập nhiều người hơn: Donald Trump hay Albert Einstein?",
        "gold_answers": ["Donald Trump"],
        "type": "comparison",
    },
    {
        "question": "Có bao nhiêu trang Wikipedia trong đồ thị tri thức?",
        "gold_answers": ["164"],
        "type": "aggregation",
    },
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def check_answer(gold_answers: list[str], predicted: str) -> bool:
    if not predicted:
        return False
    norm_pred = normalize(predicted)
    for gold in gold_answers:
        if normalize(gold) in norm_pred:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-hop QA via answer_question")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="reports/eval_multihop_qa.json")
    parser.add_argument("--dataset", default=None, help="Optional JSONL with question/gold_answers fields")
    args = parser.parse_args()

    if args.dataset:
        questions = []
        with open(args.dataset, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
    else:
        questions = MULTIHOP_QUESTIONS

    if args.limit:
        questions = questions[:args.limit]

    print(f"Evaluating {len(questions)} multi-hop questions via answer_question")

    correct = 0
    errors = 0
    details = []

    for i, q in enumerate(questions):
        question = q["question"]
        gold = q.get("gold_answers", [])

        t0 = time.time()
        try:
            result = run_agent_scaled(question)
            elapsed = time.time() - t0
            predicted = result.answer if result else ""
            hit = check_answer(gold, predicted)
            if hit:
                correct += 1
            details.append({
                "question": question,
                "type": q.get("type", "unknown"),
                "gold_answers": gold,
                "predicted": predicted[:500],
                "correct": hit,
                "latency_ms": round(elapsed * 1000, 1),
                "citations": result.citations if result else [],
            })
        except Exception as e:
            elapsed = time.time() - t0
            errors += 1
            details.append({
                "question": question,
                "type": q.get("type", "unknown"),
                "gold_answers": gold,
                "error": str(e),
                "correct": False,
                "latency_ms": round(elapsed * 1000, 1),
            })

        status = "✓" if details[-1]["correct"] else "✗"
        print(f"  [{i+1}/{len(questions)}] {status} {question[:60]}...")

    total_evaluated = len(questions) - errors
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
        "total_questions": len(questions),
        "total_evaluated": total_evaluated,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "errors": errors,
        "by_type": {k: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0} for k, v in by_type.items()},
        "details": details,
    }

    print(f"\n{'='*50}")
    print(f"Multi-hop QA Results ({total_evaluated} questions)")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Correct:   {correct}/{total_evaluated}")
    print(f"  Errors:    {errors}")
    for t, v in by_type.items():
        acc = v['correct'] / v['total'] if v['total'] > 0 else 0
        print(f"  {t}: {v['correct']}/{v['total']} ({acc:.0%})")
    print(f"{'='*50}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
