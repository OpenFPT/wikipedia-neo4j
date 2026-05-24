"""Run Baseline 2: Vector-only RAG (BM25 + dense retrieval, no KG)."""

from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient

from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "eval"
EVAL_DATA = Path(__file__).resolve().parent.parent / "data" / "viwiki_mhr" / "final" / "test.jsonl"


def vector_search(qdrant: QdrantClient, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Search Qdrant for similar paragraphs."""
    results = qdrant.search(
        collection_name="viwiki_paragraphs",
        query_vector=query_embedding,
        limit=top_k,
    )
    return [
        {
            "paragraph_id": str(hit.id),
            "text": hit.payload.get("text", "") if hit.payload else "",
            "article_title": hit.payload.get("title", "") if hit.payload else "",
            "score": hit.score,
        }
        for hit in results
    ]


def fulltext_search_neo4j(query: str, top_k: int = 5) -> list[dict]:
    """BM25-style fulltext search via Neo4j."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    results = []
    try:
        with driver.session() as session:
            records = session.run(
                """
                CALL db.index.fulltext.queryNodes('paragraph_text_ft', $query)
                YIELD node, score
                MATCH (a:Article)-[:HAS_PARAGRAPH]->(node)
                RETURN node.id AS paragraph_id,
                       node.text AS text,
                       a.title AS article_title,
                       score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                query=query,
                top_k=top_k,
            )
            for r in records:
                results.append({
                    "paragraph_id": r["paragraph_id"],
                    "text": r["text"],
                    "article_title": r["article_title"],
                    "score": r["score"],
                })
    finally:
        driver.close()
    return results


def extract_answer_from_passages(question: str, passages: list[dict]) -> str:
    """Simple extractive answer: return first passage snippet containing potential answer."""
    context = " ".join(p["text"][:300] for p in passages[:3])
    return context[:500] if context else ""


def compute_metrics(predictions: list[dict], gold: list[dict]) -> dict:
    """Compute EM and F1 for predictions vs gold answers."""
    em_count = 0
    f1_sum = 0.0

    for pred, g in zip(predictions, gold):
        pred_answer = pred.get("predicted_answer", "").strip().lower()
        gold_answer = g.get("answer", "").strip().lower()

        if pred_answer == gold_answer:
            em_count += 1

        pred_tokens = set(pred_answer.split())
        gold_tokens = set(gold_answer.split())
        if pred_tokens and gold_tokens:
            overlap = pred_tokens & gold_tokens
            precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
            recall = len(overlap) / len(gold_tokens) if gold_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            f1 = 1.0 if pred_answer == gold_answer else 0.0
        f1_sum += f1

    total = len(predictions) if predictions else 1
    return {
        "exact_match": em_count / total,
        "f1": f1_sum / total,
        "total_samples": total,
    }


def run_baseline(
    eval_path: Path = EVAL_DATA,
    output_path: Path = REPORTS_DIR / "vector_rag.json",
    top_k: int = 5,
) -> dict:
    """Run vector-only RAG baseline on test set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    predictions = []
    for sample in samples:
        question = sample["question"]

        # BM25 fulltext retrieval (no KG traversal)
        passages = fulltext_search_neo4j(question, top_k=top_k)
        predicted_answer = extract_answer_from_passages(question, passages)

        predictions.append({
            "id": sample.get("id", ""),
            "question": question,
            "predicted_answer": predicted_answer,
            "retrieved_passages": [p["paragraph_id"] for p in passages],
            "gold_answer": sample.get("answer", ""),
        })

    metrics = compute_metrics(predictions, samples)

    # Retrieval recall
    recall_hits = 0
    for pred, sample in zip(predictions, samples):
        gold_pids = set(sample.get("gold_passage_ids", []))
        retrieved_pids = set(pred["retrieved_passages"])
        if gold_pids & retrieved_pids:
            recall_hits += 1
    metrics["retrieval_recall_at_k"] = recall_hits / len(samples) if samples else 0

    result = {
        "baseline": "vector_only_rag",
        "config": {"top_k": top_k, "method": "bm25_fulltext"},
        "metrics": metrics,
        "sample_predictions": predictions[:10],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Baseline 2 complete", extra=metrics)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run vector-only RAG baseline")
    parser.add_argument("--eval-data", type=Path, default=EVAL_DATA)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "vector_rag.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    result = run_baseline(args.eval_data, args.output, args.top_k)
    print(f"Baseline 2 (Vector RAG): EM={result['metrics']['exact_match']:.3f}, "
          f"F1={result['metrics']['f1']:.3f}, "
          f"Recall@{args.top_k}={result['metrics']['retrieval_recall_at_k']:.3f}")


if __name__ == "__main__":
    main()
