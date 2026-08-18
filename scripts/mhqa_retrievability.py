"""Stage 4: Retrievability test — run WRRF retrieval on verified MHQA questions."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.logging_utils import configure_logging, get_logger

configure_logging(
    settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="mhqa_retrievability"
)
logger = get_logger(__name__)

_STOP = False


def _handle_sigint(sig, frame):
    global _STOP
    _STOP = True
    print("\nGraceful shutdown requested, finishing current batch...")


def _load_progress(progress_path: Path) -> set[str]:
    """Load set of already-tested question IDs."""
    if not progress_path.exists():
        return set()
    data = json.loads(progress_path.read_text())
    return set(data.get("processed_ids", []))


def _save_progress(progress_path: Path, processed_ids: set[str]) -> None:
    """Save progress checkpoint atomically."""
    tmp = progress_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"processed_ids": sorted(processed_ids)}, ensure_ascii=False))
    tmp.rename(progress_path)


def _answer_in_chunks(answer: str, chunks: list[dict], top_k: int = 10) -> tuple[int, bool]:
    """Check if answer appears in top-k retrieved chunks.

    Returns (rank, found) where rank is 1-based index of first hit, 0 if not found.
    """
    answer_lower = answer.lower().strip()
    for i, chunk in enumerate(chunks[:top_k]):
        chunk_text = chunk.get("chunk_text", "").lower()
        if answer_lower in chunk_text:
            return i + 1, True
    return 0, False


async def _cross_validate_question(record: dict, model: str) -> bool:
    """Cross-validate a flagged question using Claude API.

    Returns True if question is valid (keep), False if ambiguous (reject).
    """
    import anthropic

    if not settings.anthropic_api_key:
        return True

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    context_text = ""
    for ctx in record.get("context", []):
        context_text += f"\n--- {ctx['title']} ---\n"
        context_text += "\n".join(ctx.get("sentences", []))

    prompt = f"""Given the following Vietnamese Wikipedia context and question, determine:
1. Is the question answerable from the context?
2. Is there exactly one correct answer?

Context:
{context_text}

Question: {record['question']}
Expected answer: {record['answer']}

Reply in JSON format: {{"answerable": true/false, "same_answer": true/false, "reasoning": "..."}}
Only return JSON, no other text."""

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text_block = response.content[0]
        text = getattr(text_block, "text", "").strip()
        if not text:
            return True

        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        result = json.loads(text)
        return result.get("answerable", False) and result.get("same_answer", False)
    except Exception as e:
        logger.warning(f"Cross-validation failed: {e}")
        return True  # Keep on error


def run_retrievability(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    progress_path: Path,
    top_k: int = 10,
    cross_validate: bool = True,
    resume: bool = True,
):
    """Run retrievability test on verified questions."""
    from src.retrieval.fusion import _wrrf_fuse
    from src.retrieval.bm25 import run_bm25_query
    from src.retrieval.vector import vector_search
    from src.retrieval.graph import graph_search
    from src.retrieval.community import community_search

    # Load verified questions
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} verified records from {input_path}")

    # Resume support
    processed_ids = _load_progress(progress_path) if resume else set()
    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} already tested")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stats
    stats: dict = {
        "total_input": len(records),
        "top5_pass": 0,
        "rank6_10_pass": 0,
        "flagged": 0,
        "rejected": 0,
        "total_pass": 0,
        "avg_latency_ms": 0.0,
        "level_distribution": {"medium": 0, "hard": 0},
        "type_distribution": {"bridge": 0, "comparison": 0},
    }

    passed = []
    flagged_records = []
    rejected_records = []
    latencies: list[float] = []

    for record in records:
        if _STOP:
            logger.warning("Interrupted during retrievability test")
            break

        qid = record.get("_id", "")
        if qid in processed_ids:
            continue

        question = record.get("question", "")
        answer = record.get("answer", "")

        # Run WRRF hybrid retrieval
        start_time = time.time()
        try:
            bm25_results = run_bm25_query(question, top_k=top_k * 2)
            vector_results = vector_search(question, top_k=top_k * 2)
            graph_results = graph_search(question, top_k=top_k * 2)
            community_results = community_search(question, top_k=top_k * 2)

            fused = _wrrf_fuse(
                {
                    "bm25": bm25_results,
                    "vector": vector_results,
                    "graph": graph_results,
                    "community": community_results,
                },
                k=settings.wrrf_k,
            )
        except Exception as e:
            logger.warning(f"Retrieval failed for {qid}: {e}")
            fused = []

        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)

        # Check if answer in results
        rank, found = _answer_in_chunks(answer, fused, top_k=top_k)

        if found and rank <= 5:
            record["level"] = "medium"
            stats["top5_pass"] += 1
            passed.append(record)
        elif found and rank <= 10:
            record["level"] = "hard"
            stats["rank6_10_pass"] += 1
            passed.append(record)
        elif not found:
            flagged_records.append(record)
            stats["flagged"] += 1

        processed_ids.add(qid)
        _save_progress(progress_path, processed_ids)

    # Cross-validate flagged items
    if cross_validate and flagged_records and not _STOP:
        logger.info(f"Cross-validating {len(flagged_records)} flagged questions...")
        model = settings.mhqa_generation_model

        async def validate_all():
            results = []
            for record in flagged_records:
                if _STOP:
                    break
                is_valid = await _cross_validate_question(record, model)
                results.append((record, is_valid))
            return results

        validation_results = asyncio.run(validate_all())

        for record, is_valid in validation_results:
            if is_valid:
                record["level"] = "hard"
                passed.append(record)
            else:
                rejected_records.append(record)
                stats["rejected"] += 1
    else:
        rejected_records.extend(flagged_records)
        stats["rejected"] += len(flagged_records)

    # Compute final stats
    stats["total_pass"] = len(passed)
    stats["avg_latency_ms"] = sum(latencies) / max(len(latencies), 1)

    for record in passed:
        level = record.get("level", "medium")
        stats["level_distribution"][level] = stats["level_distribution"].get(level, 0) + 1
        qtype = record.get("type", "bridge")
        stats["type_distribution"][qtype] = stats["type_distribution"].get(qtype, 0) + 1

    # Write final output
    with open(output_path, "w", encoding="utf-8") as f:
        for record in passed:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    logger.info(f"Retrievability complete: {len(passed)} passed, {stats['rejected']} rejected")
    logger.info(f"Level distribution: {stats['level_distribution']}")
    logger.info(f"Avg latency: {stats['avg_latency_ms']:.1f}ms")
    logger.info(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test retrievability of MHQA questions")
    parser.add_argument("--input", type=str, default="data/mhqa/verified.jsonl", help="Input verified JSONL")
    parser.add_argument("--output", type=str, default="data/mhqa/final.jsonl", help="Output final JSONL")
    parser.add_argument("--report", type=str, default="data/mhqa/retrievability_report.json", help="Report JSON")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval check")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Start fresh")
    parser.add_argument("--no-cross-validate", action="store_true", help="Skip cross-validation of flagged items")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}. Run mhqa_qc.py first.")
        sys.exit(1)

    output_path = Path(args.output)
    report_path = Path(args.report)
    progress_path = output_path.parent / ".retrieve_progress.json"

    run_retrievability(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        progress_path=progress_path,
        top_k=args.top_k,
        cross_validate=not args.no_cross_validate,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
