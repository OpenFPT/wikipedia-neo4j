"""Evaluate GraphRAG via Claude Haiku + MCP tool definitions against ViQuAD2.

Usage:
    uv run python scripts/eval_mcp_claude.py --limit 100
    uv run python scripts/eval_mcp_claude.py --limit 200 --model claude-haiku-4-5-20251001
    uv run python scripts/eval_mcp_claude.py --dataset data/viquad2/validation.jsonl --output reports/eval_mcp.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from pathlib import Path

import anthropic

TOOL_DEFINITIONS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Use for factual questions needing 1-2 facts from Vietnamese Wikipedia. "
            "Examples: 'Ai sáng lập Đảng Cộng sản Việt Nam?', 'Hà Nội có bao nhiêu quận?' "
            "Do NOT use for questions requiring reasoning across multiple facts — use explore_entity or find_path. "
            "Returns ranked text passages with source page links."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question in Vietnamese"},
                "top_k": {"type": "integer", "description": "Number of results (1-20)", "default": 5},
            },
            "required": ["question"],
        },
    },
    {
        "name": "explore_entity",
        "description": (
            "Use when you know an entity name and want to see what connects to it. "
            "Examples: explore 'Hồ Chí Minh' to find related people, places, events. "
            "Do NOT use for searching — use search_knowledge_base first to find entity names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_name": {"type": "string", "description": "Entity name in Vietnamese"},
                "depth": {"type": "integer", "description": "Traversal depth (1-2)", "default": 1},
            },
            "required": ["entity_name"],
        },
    },
    {
        "name": "find_path",
        "description": (
            "Use for questions asking HOW two things are connected or related. "
            "Examples: 'Mối quan hệ giữa Phạm Văn Đồng và Hồ Chí Minh?', 'X liên quan đến Y thế nào?' "
            "Requires knowing both entity names — search first if unsure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_a": {"type": "string", "description": "Starting entity name"},
                "entity_b": {"type": "string", "description": "Target entity name"},
                "max_hops": {"type": "integer", "description": "Max path length (2-6)", "default": 5},
            },
            "required": ["entity_a", "entity_b"],
        },
    },
    {
        "name": "get_community_summary",
        "description": (
            "Use for broad topic overviews before diving into specifics. "
            "Returns a summary of related concepts clustered around a topic. "
            "Examples: get overview of 'Chiến tranh Việt Nam' or 'Văn học Việt Nam'. "
            "Do NOT use for specific factual questions — use search_knowledge_base instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic or entity name"},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "source_trace",
        "description": (
            "Use after search_knowledge_base to get more context around a passage. "
            "Given a chunk_id from search results, returns the full source page and neighboring passages. "
            "Use when a search result is relevant but you need more surrounding context to answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string", "description": "Chunk ID from search results"},
            },
            "required": ["chunk_id"],
        },
    },
]

SYSTEM_PROMPT = (
    "You are a Vietnamese Wikipedia QA assistant with access to a knowledge graph. "
    "Given a question, use the available tools to find relevant information, then provide "
    "a concise answer in Vietnamese. Answer with just the factual answer, no explanation needed. "
    "If you cannot find the answer, say 'Không tìm thấy thông tin'."
)


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison: lowercase, remove punctuation/articles."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    return text.strip()


def compute_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, gold: str) -> float:
    """Exact match after normalization."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def compute_best_metrics(prediction: str, gold_answers: list[str]) -> dict:
    """Compute best F1 and EM across all gold answers."""
    if not gold_answers:
        return {"f1": 0.0, "em": 0.0}
    best_f1 = max(compute_f1(prediction, g) for g in gold_answers)
    best_em = max(compute_em(prediction, g) for g in gold_answers)
    return {"f1": best_f1, "em": best_em}


def compute_faithfulness(answer: str, retrieved_texts: list[str]) -> float:
    """Entity-overlap faithfulness: fraction of answer tokens found in context."""
    if not answer.strip() or not retrieved_texts:
        return 0.0
    answer_tokens = set(normalize_answer(answer).split())
    context = normalize_answer(" ".join(retrieved_texts))
    context_tokens = set(context.split())
    if not answer_tokens:
        return 1.0
    grounded = answer_tokens & context_tokens
    return len(grounded) / len(answer_tokens)


def compute_efficiency(tool_calls: int) -> float:
    """Efficiency score: fewer calls = better. 1 call = 1.0, diminishes."""
    if tool_calls == 0:
        return 0.0
    return 1.0 / tool_calls


def detect_refusal(answer: str) -> bool:
    """Check if the model refused to answer (appropriate for unsolvable questions)."""
    refusal_phrases = [
        "không tìm thấy",
        "không có thông tin",
        "không thể tìm",
        "không có dữ liệu",
        "không rõ",
        "không biết",
    ]
    lower = answer.lower()
    return any(p in lower for p in refusal_phrases)


def load_dataset(path: Path, limit: int | None = None) -> list[dict]:
    """Load ViQuAD2 JSONL dataset."""
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
            samples.append({
                "id": item["id"],
                "question": item["question"],
                "gold_answers": gold,
                "is_unsolvable": item.get("is_unsolvable", False),
            })
            if limit and len(samples) >= limit:
                break
    return samples


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call against the live GraphRAG system."""
    from src.mcp_pkg.tools import register_tools

    class FakeMCP:
        def __init__(self):
            self._tools = {}
            self._resources = {}

        def tool(self):
            def decorator(fn):
                self._tools[fn.__name__] = fn
                return fn
            return decorator

        def resource(self, uri: str):
            def decorator(fn):
                self._resources[uri] = fn
                return fn
            return decorator

    mcp = FakeMCP()
    register_tools(mcp)

    if tool_name not in mcp._tools:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        result = mcp._tools[tool_name](**tool_input)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def run_single_question(
    client: anthropic.Anthropic,
    question: str,
    model: str,
    max_tool_rounds: int = 3,
) -> dict:
    """Run a single question through Claude with tool use, capturing trajectory."""
    messages = [{"role": "user", "content": question}]
    tool_calls_count = 0
    trajectory = []
    retrieved_texts = []
    empty_calls = 0
    t0 = time.perf_counter()

    for _ in range(max_tool_rounds + 1):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            answer = ""
            for block in response.content:
                if block.type == "text":
                    answer += block.text
            return {
                "answer": answer.strip(),
                "tool_calls": tool_calls_count,
                "trajectory": trajectory,
                "retrieved_texts": retrieved_texts,
                "empty_calls": empty_calls,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls_count += 1
                    result_str = execute_tool(block.name, block.input)
                    result_data = json.loads(result_str) if result_str else {}
                    is_empty = (
                        result_data.get("total", 1) == 0
                        or result_data.get("error")
                        or (result_data.get("results") == [])
                        or (result_data.get("neighbors") == [])
                    )
                    if is_empty:
                        empty_calls += 1
                    # Collect retrieved text for faithfulness
                    for r in result_data.get("results", []):
                        if r.get("text"):
                            retrieved_texts.append(r["text"])
                    if result_data.get("chunk_text"):
                        retrieved_texts.append(result_data["chunk_text"])
                    trajectory.append({
                        "tool": block.name,
                        "args": block.input,
                        "result_len": len(result_str),
                        "is_empty": is_empty,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str[:4000],
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            answer = ""
            for block in response.content:
                if block.type == "text":
                    answer += block.text
            return {
                "answer": answer.strip(),
                "tool_calls": tool_calls_count,
                "trajectory": trajectory,
                "retrieved_texts": retrieved_texts,
                "empty_calls": empty_calls,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

    return {
        "answer": "",
        "tool_calls": tool_calls_count,
        "trajectory": trajectory,
        "retrieved_texts": retrieved_texts,
        "empty_calls": empty_calls,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "input_tokens": 0,
        "output_tokens": 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG via Claude + tools")
    parser.add_argument("--dataset", default="data/viquad2/validation.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--output", default="reports/eval_mcp_claude.json")
    parser.add_argument("--max-tool-rounds", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if not api_key:
        print("ERROR: Set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY environment variable")
        return

    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client_kwargs["default_headers"] = {
        "User-Agent": "Claude Code/2.1.152",
        "x-anthropic-billing-header": "cc_version=2.1.152.9fe; cc_entrypoint=claude-vscode; cch=868f7;",
    }
    client = anthropic.Anthropic(**client_kwargs)
    dataset = load_dataset(Path(args.dataset), limit=args.limit)
    print(f"Loaded {len(dataset)} questions from {args.dataset}")
    print(f"Model: {args.model}, max tool rounds: {args.max_tool_rounds}")

    results = []
    total_f1 = 0.0
    total_em = 0.0
    total_faithfulness = 0.0
    total_efficiency = 0.0
    total_tool_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_empty_calls = 0
    refusals = 0
    errors = 0

    for i, sample in enumerate(dataset):
        try:
            result = run_single_question(
                client, sample["question"], args.model, args.max_tool_rounds
            )
            metrics = compute_best_metrics(result["answer"], sample["gold_answers"])
            faith = compute_faithfulness(result["answer"], result.get("retrieved_texts", []))
            eff = compute_efficiency(result["tool_calls"])
            is_refusal = detect_refusal(result["answer"])
            is_unsolvable = sample.get("is_unsolvable", False)

            result.update(metrics)
            result["faithfulness"] = round(faith, 4)
            result["efficiency"] = round(eff, 4)
            result["is_refusal"] = is_refusal
            result["is_unsolvable"] = is_unsolvable
            result["first_tool"] = result["trajectory"][0]["tool"] if result.get("trajectory") else None
            result["id"] = sample["id"]
            result["question"] = sample["question"]
            result["gold_answers"] = sample["gold_answers"]
            # Don't store large retrieved_texts in output
            result.pop("retrieved_texts", None)
            results.append(result)

            total_f1 += metrics["f1"]
            total_em += metrics["em"]
            total_faithfulness += faith
            total_efficiency += eff
            total_tool_calls += result["tool_calls"]
            total_empty_calls += result.get("empty_calls", 0)
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            if is_refusal:
                refusals += 1
        except Exception as e:
            errors += 1
            results.append({
                "id": sample["id"],
                "question": sample["question"],
                "error": str(e),
                "f1": 0.0,
                "em": 0.0,
            })

        if (i + 1) % 10 == 0:
            avg_f1 = total_f1 / (i + 1 - errors) if (i + 1 - errors) > 0 else 0
            print(f"  [{i+1}/{len(dataset)}] F1={avg_f1:.3f} | tools={total_tool_calls} | errors={errors}")

    n = len(dataset) - errors
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "total_questions": len(dataset),
        "evaluated": n,
        "errors": errors,
        "avg_f1": round(total_f1 / n, 4) if n else 0,
        "avg_em": round(total_em / n, 4) if n else 0,
        "avg_faithfulness": round(total_faithfulness / n, 4) if n else 0,
        "avg_efficiency": round(total_efficiency / n, 4) if n else 0,
        "avg_tool_calls": round(total_tool_calls / n, 2) if n else 0,
        "total_empty_calls": total_empty_calls,
        "refusals": refusals,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(
            total_input_tokens * 0.80 / 1_000_000 + total_output_tokens * 4.0 / 1_000_000, 4
        ),
    }

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Model:          {summary['model']}")
    print(f"  Questions:      {summary['evaluated']}/{summary['total_questions']}")
    print(f"  Avg F1:         {summary['avg_f1']:.4f}")
    print(f"  Avg EM:         {summary['avg_em']:.4f}")
    print(f"  Avg Faithful:   {summary['avg_faithfulness']:.4f}")
    print(f"  Avg Efficiency: {summary['avg_efficiency']:.4f}")
    print(f"  Avg tool calls: {summary['avg_tool_calls']:.2f}")
    print(f"  Empty calls:    {summary['total_empty_calls']}")
    print(f"  Refusals:       {summary['refusals']}")
    print(f"  Total tokens:   {summary['total_input_tokens']} in / {summary['total_output_tokens']} out")
    print(f"  Est. cost:      ${summary['estimated_cost_usd']:.4f}")
    print(f"  Errors:         {summary['errors']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
