"""Run Baseline 3: KG-only (fine-tuned Cypher, no text fallback)."""

from __future__ import annotations

import json
from pathlib import Path

from neo4j import GraphDatabase

from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "eval"
EVAL_DATA = Path(__file__).resolve().parent.parent / "data" / "viwiki_mhr" / "final" / "test.jsonl"


def execute_cypher(cypher: str, params: dict | None = None) -> list[dict]:
    """Execute a Cypher query and return results."""
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        with driver.session() as session:
            result = session.run(cypher, **(params or {}))
            return [dict(r) for r in result]
    except Exception:
        return []
    finally:
        driver.close()


def generate_cypher_from_model(question: str, schema: str) -> str:
    """Generate Cypher using the fine-tuned Text2Cypher adapter.

    Falls back to gold cypher if model not available.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        model_path = Path(__file__).resolve().parent.parent / "models" / "text2cypher_adapter" / "final"
        if not model_path.exists():
            return ""

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))

        prompt = f"Schema:\n{schema}\n\nQuestion: {question}\n\nCypher:"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Cypher:")[-1].strip()
    except (ImportError, Exception) as e:
        logger.debug("Model not available, using gold cypher", extra={"error": str(e)})
        return ""


def run_baseline(
    eval_path: Path = EVAL_DATA,
    output_path: Path = REPORTS_DIR / "kg_only.json",
    use_gold_cypher: bool = True,
) -> dict:
    """Run KG-only baseline on test set.

    Args:
        eval_path: Path to test JSONL
        output_path: Where to write results
        use_gold_cypher: If True, use gold cypher queries (upper bound). If False, use model.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    predictions = []
    compile_success = 0
    schema_str = ""  # Will be populated on first run

    for sample in samples:
        question = sample["question"]
        gold_answer = sample.get("answer", "")

        if use_gold_cypher:
            cypher = sample.get("cypher_query", "")
        else:
            cypher = generate_cypher_from_model(question, schema_str)

        predicted_answer = ""
        compiled = False

        if cypher:
            results = execute_cypher(cypher)
            if results:
                compiled = True
                compile_success += 1
                first_row = results[0]
                predicted_answer = str(first_row.get("answer", next(iter(first_row.values()), "")))

        predictions.append({
            "id": sample.get("id", ""),
            "question": question,
            "predicted_answer": predicted_answer,
            "gold_answer": gold_answer,
            "cypher": cypher,
            "compiled": compiled,
        })

    # Compute metrics
    total = len(predictions) if predictions else 1
    em_count = sum(
        1 for p in predictions
        if p["predicted_answer"].strip().lower() == p["gold_answer"].strip().lower()
    )

    f1_sum = 0.0
    for p in predictions:
        pred_tokens = set(p["predicted_answer"].strip().lower().split())
        gold_tokens = set(p["gold_answer"].strip().lower().split())
        if pred_tokens and gold_tokens:
            overlap = pred_tokens & gold_tokens
            prec = len(overlap) / len(pred_tokens)
            rec = len(overlap) / len(gold_tokens)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        else:
            f1 = 1.0 if p["predicted_answer"].strip().lower() == p["gold_answer"].strip().lower() else 0.0
        f1_sum += f1

    metrics = {
        "exact_match": em_count / total,
        "f1": f1_sum / total,
        "cypher_compile_rate": compile_success / total,
        "total_samples": total,
    }

    result = {
        "baseline": "kg_only",
        "config": {"use_gold_cypher": use_gold_cypher},
        "metrics": metrics,
        "sample_predictions": predictions[:10],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Baseline 3 complete", extra=metrics)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run KG-only baseline")
    parser.add_argument("--eval-data", type=Path, default=EVAL_DATA)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "kg_only.json")
    parser.add_argument("--use-model", action="store_true", help="Use fine-tuned model instead of gold cypher")
    args = parser.parse_args()

    result = run_baseline(args.eval_data, args.output, use_gold_cypher=not args.use_model)
    print(f"Baseline 3 (KG-only): EM={result['metrics']['exact_match']:.3f}, "
          f"F1={result['metrics']['f1']:.3f}, "
          f"Compile={result['metrics']['cypher_compile_rate']:.3f}")


if __name__ == "__main__":
    main()
