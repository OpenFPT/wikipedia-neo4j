"""Predict missing LINKS_TO relationships between Wikipedia pages using LLM in-context learning."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types

from src.config import settings
from src.infrastructure.llm import _client_pool, _is_retryable_gemini_error
from src.logging_utils import configure_logging, get_logger
from src.infrastructure.neo4j_client import Neo4jClient

configure_logging(settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="predict_links")
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

_CANDIDATE_QUERY = """
MATCH (a:Page)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(:Chunk)<-[:HAS_CHUNK]-(b:Page)
WHERE a <> b
  AND NOT (a)-[:LINKS_TO]->(b)
  AND id(a) < id(b)
WITH a, b, collect(DISTINCT e.name) AS shared_entities
WHERE size(shared_entities) >= $min_shared
RETURN a.title AS page_a_title,
       a.id AS page_a_id,
       a.summary AS page_a_summary,
       b.title AS page_b_title,
       b.id AS page_b_id,
       b.summary AS page_b_summary,
       shared_entities
ORDER BY size(shared_entities) DESC
LIMIT $limit
"""

_PAGE_ENTITIES_QUERY = """
MATCH (p:Page {id: $page_id})-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(e:Entity)
RETURN collect(DISTINCT e.name)[..20] AS entities
"""

_PAGE_LINKS_QUERY = """
MATCH (p:Page {id: $page_id})-[:LINKS_TO]->(target:Page)
RETURN target.title AS title
LIMIT 5
"""


def _fetch_candidates(client: Neo4jClient, min_shared: int, limit: int) -> list[dict]:
    """Fetch candidate page pairs that share entities but lack LINKS_TO."""
    with client.session() as session:
        result = session.run(_CANDIDATE_QUERY, min_shared=min_shared, limit=limit)
        return [dict(record) for record in result]


def _fetch_page_context(client: Neo4jClient, page_id: str) -> dict:
    """Fetch entities and existing links for a page."""
    with client.session() as session:
        ent_result = session.run(_PAGE_ENTITIES_QUERY, page_id=page_id)
        ent_record = ent_result.single()
        entities = ent_record["entities"] if ent_record else []

        link_result = session.run(_PAGE_LINKS_QUERY, page_id=page_id)
        links = [record["title"] for record in link_result]

    return {"entities": entities, "links": links}


# ---------------------------------------------------------------------------
# LLM prediction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a Wikipedia link prediction expert. Given structural context about two Wikipedia pages, determine whether Page A should have a hyperlink (LINKS_TO) to Page B.

Consider:
- Topical relevance: Are the pages about related subjects?
- Shared entities: Do they discuss the same people, places, or concepts?
- Link patterns: Based on existing links, would this link be natural?
- Wikipedia conventions: Would a reader of Page A benefit from a link to Page B?

Respond with EXACTLY this JSON format (no markdown fences):
{"decision": "YES" or "NO", "confidence": 0.0 to 1.0, "reason": "brief explanation"}"""


def _build_knowledge_prompt(candidate: dict, ctx_a: dict, ctx_b: dict) -> str:
    """Build the knowledge prompt with structural context for a candidate pair."""
    page_a_summary = (candidate.get("page_a_summary") or "N/A")[:300]
    page_b_summary = (candidate.get("page_b_summary") or "N/A")[:300]
    shared = candidate["shared_entities"]

    lines = [
        "## Page A",
        f"Title: {candidate['page_a_title']}",
        f"Summary: {page_a_summary}",
        f"Entities mentioned: {', '.join(ctx_a['entities'][:15])}",
        f"Existing links from Page A (sample): {', '.join(ctx_a['links']) or 'none'}",
        "",
        "## Page B",
        f"Title: {candidate['page_b_title']}",
        f"Summary: {page_b_summary}",
        f"Entities mentioned: {', '.join(ctx_b['entities'][:15])}",
        f"Existing links from Page B (sample): {', '.join(ctx_b['links']) or 'none'}",
        "",
        f"## Shared entities ({len(shared)}): {', '.join(shared[:10])}",
        "",
        "Should Page A link to Page B?",
    ]
    return "\n".join(lines)


def _parse_llm_response(text: str) -> dict | None:
    """Parse LLM JSON response into a prediction dict."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r"\{[^}]+\}", text)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    decision = str(parsed.get("decision", "")).upper().strip()
    confidence = float(parsed.get("confidence", 0))
    reason = str(parsed.get("reason", ""))

    if decision not in ("YES", "NO"):
        return None

    return {"decision": decision, "confidence": confidence, "reason": reason}


def _predict_batch(
    candidates: list[dict],
    contexts: list[tuple[dict, dict]],
    max_retries: int = 3,
) -> list[dict | None]:
    """Run LLM predictions for a batch of candidates."""
    clients = _client_pool()
    results: list[dict | None] = []

    for candidate, (ctx_a, ctx_b) in zip(candidates, contexts):
        user_prompt = _build_knowledge_prompt(candidate, ctx_a, ctx_b)
        prediction = None

        for attempt in range(max_retries):
            client = random.choice(clients)
            try:
                resp = client.models.generate_content(
                    model=settings.gemini_model_text,
                    contents=f"{_SYSTEM_PROMPT}\n\n{user_prompt}",
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=256,
                    ),
                )
                raw_text = resp.text or ""
                prediction = _parse_llm_response(raw_text)
                if prediction is not None:
                    break
                logger.warning(
                    "Failed to parse LLM response, retrying",
                    extra={"attempt": attempt + 1, "raw": raw_text[:200]},
                )
            except Exception as exc:
                logger.warning(
                    "LLM call failed",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )
                if _is_retryable_gemini_error(exc):
                    delay = min(2 ** (attempt + 1), 8) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                break

        results.append(prediction)
        # Small delay between calls to respect rate limits
        time.sleep(0.5)

    return results


# ---------------------------------------------------------------------------
# Neo4j write
# ---------------------------------------------------------------------------

_CREATE_LINK_QUERY = """
UNWIND $pairs AS pair
MATCH (a:Page {id: pair.page_a_id})
MATCH (b:Page {id: pair.page_b_id})
CREATE (a)-[:LINKS_TO {predicted: true, confidence: pair.confidence, predicted_at: datetime($ts)}]->(b)
"""


def _write_predictions(client: Neo4jClient, accepted: list[dict]) -> int:
    """Write accepted predictions as LINKS_TO relationships."""
    if not accepted:
        return 0

    ts = datetime.now(timezone.utc).isoformat()
    pairs = [
        {
            "page_a_id": p["page_a_id"],
            "page_b_id": p["page_b_id"],
            "confidence": p["confidence"],
        }
        for p in accepted
    ]

    with client.session() as session:
        session.run(_CREATE_LINK_QUERY, pairs=pairs, ts=ts)

    logger.info("Wrote predicted LINKS_TO relationships", extra={"count": len(pairs)})
    return len(pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict missing LINKS_TO relationships using LLM in-context learning."
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of candidate pairs to evaluate")
    parser.add_argument(
        "--min-shared-entities", type=int, default=2, help="Minimum shared entities to consider a pair"
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.7, help="Minimum confidence to accept prediction"
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Neo4j, just report predictions")
    parser.add_argument("--output", type=str, default=None, help="Save predictions to JSONL file")
    parser.add_argument("--batch-size", type=int, default=10, help="Pairs per batch")
    args = parser.parse_args()

    logger.info(
        "Starting link prediction",
        extra={
            "limit": args.limit,
            "min_shared_entities": args.min_shared_entities,
            "confidence_threshold": args.confidence_threshold,
            "dry_run": args.dry_run,
        },
    )

    client = Neo4jClient()
    try:
        client.verify_connectivity()
    except Exception as exc:
        logger.error("Cannot connect to Neo4j", extra={"error": str(exc)})
        sys.exit(1)

    # Fetch candidates
    logger.info("Fetching candidate pairs...")
    candidates = _fetch_candidates(client, args.min_shared_entities, args.limit)
    logger.info("Found candidate pairs", extra={"count": len(candidates)})

    if not candidates:
        logger.info("No candidate pairs found, exiting.")
        client.close()
        return

    # Process in batches
    all_predictions: list[dict] = []
    accepted: list[dict] = []

    for batch_start in range(0, len(candidates), args.batch_size):
        batch = candidates[batch_start : batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        logger.info(f"Processing batch {batch_num}", extra={"size": len(batch)})

        # Fetch context for each candidate in the batch
        contexts: list[tuple[dict, dict]] = []
        for cand in batch:
            ctx_a = _fetch_page_context(client, cand["page_a_id"])
            ctx_b = _fetch_page_context(client, cand["page_b_id"])
            contexts.append((ctx_a, ctx_b))

        # Run LLM predictions
        predictions = _predict_batch(batch, contexts)

        for cand, pred in zip(batch, predictions):
            record = {
                "page_a_id": cand["page_a_id"],
                "page_a_title": cand["page_a_title"],
                "page_b_id": cand["page_b_id"],
                "page_b_title": cand["page_b_title"],
                "shared_entities": cand["shared_entities"],
                "prediction": pred,
            }
            all_predictions.append(record)

            if pred and pred["decision"] == "YES" and pred["confidence"] >= args.confidence_threshold:
                accepted.append({
                    "page_a_id": cand["page_a_id"],
                    "page_a_title": cand["page_a_title"],
                    "page_b_id": cand["page_b_id"],
                    "page_b_title": cand["page_b_title"],
                    "confidence": pred["confidence"],
                    "reason": pred["reason"],
                })

    # Summary
    total = len(all_predictions)
    yes_count = sum(1 for p in all_predictions if p["prediction"] and p["prediction"]["decision"] == "YES")
    no_count = sum(1 for p in all_predictions if p["prediction"] and p["prediction"]["decision"] == "NO")
    failed = sum(1 for p in all_predictions if p["prediction"] is None)

    logger.info(
        "Prediction summary",
        extra={
            "total": total,
            "yes": yes_count,
            "no": no_count,
            "failed_parse": failed,
            "accepted_above_threshold": len(accepted),
        },
    )
    print("\n--- Link Prediction Results ---")
    print(f"Total pairs evaluated: {total}")
    print(f"  YES: {yes_count}")
    print(f"  NO:  {no_count}")
    print(f"  Parse failures: {failed}")
    print(f"  Accepted (confidence >= {args.confidence_threshold}): {len(accepted)}")

    # Write output JSONL
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in all_predictions:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Predictions saved", extra={"path": str(output_path)})
        print(f"Predictions saved to: {output_path}")

    # Write to Neo4j
    if accepted and not args.dry_run:
        written = _write_predictions(client, accepted)
        print(f"Created {written} predicted LINKS_TO relationships in Neo4j.")
    elif accepted and args.dry_run:
        print("\n[DRY RUN] Would create these LINKS_TO relationships:")
        for a in accepted:
            print(f"  {a['page_a_title']} -> {a['page_b_title']} (confidence={a['confidence']:.2f}, reason={a['reason']})")
    else:
        print("No predictions met the confidence threshold.")

    client.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
