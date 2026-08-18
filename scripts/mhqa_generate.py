"""Stage 2: Generate multi-hop QA pairs using Claude API from walk structures."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.logging_utils import configure_logging, get_logger

configure_logging(settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="mhqa_generate")
logger = get_logger(__name__)

_STOP = False


def _handle_sigint(sig, frame):
    global _STOP
    _STOP = True
    print("\nGraceful shutdown requested, finishing current batch...")


def _load_seeds(seeds_path: Path) -> list[dict]:
    """Load seed examples from JSONL."""
    seeds = []
    with open(seeds_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    logger.info(f"Loaded {len(seeds)} seed examples from {seeds_path}")
    return seeds


def _load_progress(progress_path: Path) -> set[str]:
    """Load set of already-processed walk_ids."""
    if not progress_path.exists():
        return set()
    data = json.loads(progress_path.read_text())
    return set(data.get("processed_walk_ids", []))


def _save_progress(progress_path: Path, processed_ids: set[str]) -> None:
    """Save progress checkpoint atomically."""
    tmp = progress_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"processed_walk_ids": sorted(processed_ids)}, ensure_ascii=False))
    tmp.rename(progress_path)


def _select_few_shot(seeds: list[dict], walk: dict, n: int = 3) -> list[dict]:
    """Select n seed examples matching the walk type and hop count."""
    matching = [s for s in seeds if s.get("type") == walk["type"] and s.get("num_hops") == walk["num_hops"]]
    if len(matching) < n:
        matching = seeds  # fallback to all seeds
    return random.sample(matching, min(n, len(matching)))


def _format_seed_example(seed: dict) -> str:
    """Format a seed example for the prompt."""
    return json.dumps(
        {
            "question": seed["question"],
            "answer": seed["answer"],
            "supporting_facts": seed["supporting_facts"],
            "decomposition": seed["decomposition"],
            "type": seed["type"],
            "num_hops": seed["num_hops"],
        },
        ensure_ascii=False,
        indent=2,
    )


def _format_walk_context(walk: dict) -> str:
    """Format walk data for the prompt."""
    parts = []
    parts.append(f"Walk type: {walk['type']}, Hops: {walk['num_hops']}")
    parts.append(f"Bridge entities: {', '.join(walk['bridge_entities'])}")
    parts.append(f"Entity types: {', '.join(walk['entity_types'])}")
    parts.append("")

    for i, page in enumerate(walk["pages"], 1):
        parts.append(f"--- Page {i}: {page['title']} (page_id: {page['page_id']}) ---")
        for chunk in page["chunks"]:
            parts.append(f"Chunk ID: {chunk['chunk_id']}")
            parts.append("Sentences:")
            for j, sent in enumerate(chunk["sentences"]):
                parts.append(f"  [{j}] {sent}")
        parts.append("")

    return "\n".join(parts)


def _build_prompt(walks_batch: list[dict], seeds: list[dict]) -> str:
    """Build the generation prompt for a batch of walks."""
    few_shot_examples = []
    for walk in walks_batch[:1]:  # Use seeds matching first walk's type
        few_shot_examples = _select_few_shot(seeds, walk, n=3)
        break

    prompt_parts = [
        "You are generating multi-hop Vietnamese Wikipedia questions.",
        "",
        "Given the context from two Wikipedia pages connected by a bridge entity,",
        "write a natural multi-hop question that requires reading BOTH pages to answer.",
        "",
        "Rules:",
        "- The question must be in Vietnamese",
        "- The answer must be a short span (1-5 words) extractable from the context",
        "- Provide decomposition into sub-questions",
        "- Mark which sentences are supporting facts (by sent_id index within the page's sentences array)",
        "- Question should NOT mention page titles directly",
        "- Each supporting_fact needs: title (page title), sent_id (sentence index), page_id",
        "- Each decomposition entry needs: sub_question, sub_answer, page_id",
        "",
        "## Few-shot examples:",
        "",
    ]

    for i, ex in enumerate(few_shot_examples, 1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(_format_seed_example(ex))
        prompt_parts.append("")

    prompt_parts.append(f"## Now generate {len(walks_batch)} QA pair(s) for these contexts:")
    prompt_parts.append("")

    for i, walk in enumerate(walks_batch, 1):
        prompt_parts.append(f"### Context {i} (walk_id: {walk['walk_id']}):")
        prompt_parts.append(_format_walk_context(walk))

    prompt_parts.extend([
        "",
        "## Output format:",
        "",
        "Return a JSON array with one object per context. Each object must have:",
        '- "walk_id": the walk_id from above',
        '- "question": Vietnamese multi-hop question (string)',
        '- "answer": short answer span (string)',
        '- "supporting_facts": array of {"title": str, "sent_id": int, "page_id": str}',
        '- "decomposition": array of {"sub_question": str, "sub_answer": str, "page_id": str}',
        '- "type": same as walk type',
        '- "num_hops": same as walk num_hops',
        "",
        "Return ONLY the JSON array, no other text.",
    ])

    return "\n".join(prompt_parts)


async def _call_claude(client, prompt: str, model: str) -> list[dict] | None:
    """Call Claude API and parse response as JSON array."""
    import anthropic

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text_block = response.content[0]
        text = getattr(text_block, "text", "").strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Claude response as JSON: {e}")
        return None
    except anthropic.RateLimitError:
        logger.warning("Rate limited, will retry after backoff")
        raise
    except anthropic.APIError as e:
        logger.error(f"Claude API error: {e}")
        return None


async def _generate_batch(
    client,
    walks_batch: list[dict],
    seeds: list[dict],
    model: str,
    max_retries: int = 3,
) -> list[dict]:
    """Generate QA pairs for a batch of walks with retry."""
    prompt = _build_prompt(walks_batch, seeds)

    for attempt in range(max_retries):
        try:
            results = await _call_claude(client, prompt, model)
            if results and isinstance(results, list):
                return results
            logger.warning(f"Invalid response on attempt {attempt + 1}/{max_retries}")
        except Exception as e:
            wait_time = 2 ** (attempt + 1)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    return []


def _format_qa_record(raw: dict, walk: dict) -> dict | None:
    """Format a raw generated QA into the output schema."""
    try:
        # Build context from walk pages
        context = []
        for page in walk["pages"]:
            sentences = []
            for chunk in page["chunks"]:
                sentences.extend(chunk["sentences"])
            context.append({
                "title": page["title"],
                "sentences": sentences,
                "page_id": page["page_id"],
            })

        record = {
            "_id": None,  # Assigned in QC stage
            "question": raw["question"],
            "answer": raw["answer"],
            "supporting_facts": raw.get("supporting_facts", []),
            "context": context,
            "type": raw.get("type", walk["type"]),
            "level": "medium",  # Assigned in retrievability stage
            "source": "kg-walk+claude",
            "num_hops": raw.get("num_hops", walk["num_hops"]),
            "is_impossible": False,
            "plausible_answer": None,
            "decomposition": raw.get("decomposition", []),
            "corpus_snapshot": "viwiki-20260523",
            "schema_version": "1.0",
            "walk_id": walk["walk_id"],  # Keep for traceability
        }
        return record
    except (KeyError, TypeError) as e:
        logger.warning(f"Failed to format QA record: {e}")
        return None


async def run_generation(
    walks_path: Path,
    seeds_path: Path,
    output_path: Path,
    progress_path: Path,
    batch_size: int,
    concurrency: int,
    model: str,
    resume: bool = True,
):
    """Run the generation pipeline."""
    import anthropic

    if not settings.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY not set. Cannot proceed.")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    # Load inputs
    seeds = _load_seeds(seeds_path)
    walks = []
    with open(walks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                walks.append(json.loads(line))
    logger.info(f"Loaded {len(walks)} walks from {walks_path}")

    # Resume support
    processed_ids = _load_progress(progress_path) if resume else set()
    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} walks already processed")

    remaining_walks = [w for w in walks if w["walk_id"] not in processed_ids]
    logger.info(f"Remaining walks to process: {len(remaining_walks)}")

    # Open output file in append mode
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create batches
    batches = [remaining_walks[i : i + batch_size] for i in range(0, len(remaining_walks), batch_size)]
    logger.info(f"Processing {len(batches)} batches of size {batch_size}")

    semaphore = asyncio.Semaphore(concurrency)
    total_generated = 0

    async def process_batch(batch: list[dict]) -> list[dict]:
        nonlocal total_generated
        if _STOP:
            return []
        async with semaphore:
            if _STOP:
                return []
            results = await _generate_batch(client, batch, seeds, model)
            records = []
            walk_map = {w["walk_id"]: w for w in batch}

            for raw in results:
                walk_id = raw.get("walk_id")
                if walk_id and walk_id in walk_map:
                    record = _format_qa_record(raw, walk_map[walk_id])
                    if record:
                        records.append(record)

            # Write results immediately
            if records:
                with open(output_path, "a", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total_generated += len(records)

            # Update progress
            for w in batch:
                processed_ids.add(w["walk_id"])
            _save_progress(progress_path, processed_ids)

            logger.info(
                f"Batch done: {len(records)}/{len(batch)} generated "
                f"(total: {total_generated}, progress: {len(processed_ids)}/{len(walks)})"
            )
            return records

    # Process batches with concurrency
    tasks = [process_batch(batch) for batch in batches]
    await asyncio.gather(*tasks)

    logger.info(f"Generation complete. Total QA pairs generated: {total_generated}")
    logger.info(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate MHQA pairs using Claude API")
    parser.add_argument("--walks", type=str, default="data/mhqa/walks.jsonl", help="Input walks JSONL")
    parser.add_argument("--seeds", type=str, default="seeds.jsonl", help="Seed examples JSONL")
    parser.add_argument("--output", type=str, default="data/mhqa/raw_questions.jsonl", help="Output JSONL")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Start fresh")
    parser.add_argument("--batch-size", type=int, default=None, help="Walks per Claude call")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel API calls")
    parser.add_argument("--model", type=str, default=None, help="Claude model to use")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    walks_path = Path(args.walks)
    if not walks_path.exists():
        logger.error(f"Walks file not found: {walks_path}. Run mhqa_extract_walks.py first.")
        sys.exit(1)

    seeds_path = Path(args.seeds)
    if not seeds_path.exists():
        logger.error(f"Seeds file not found: {seeds_path}")
        sys.exit(1)

    output_path = Path(args.output)
    progress_path = output_path.parent / ".generate_progress.json"

    batch_size = args.batch_size or settings.mhqa_batch_size
    concurrency = args.concurrency or settings.mhqa_concurrency
    model = args.model or settings.mhqa_generation_model

    asyncio.run(
        run_generation(
            walks_path=walks_path,
            seeds_path=seeds_path,
            output_path=output_path,
            progress_path=progress_path,
            batch_size=batch_size,
            concurrency=concurrency,
            model=model,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
