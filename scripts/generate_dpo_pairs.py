"""Build DPO preference pairs: chosen (valid Cypher) vs rejected (hallucinated/fluff)."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from src.logging_utils import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "finetune"


@dataclass
class DPOPair:
    """A single DPO training pair."""

    prompt: str
    chosen: str
    rejected: str


FLUFF_PREFIXES = [
    "Sure! Here's the Cypher query you need:\n\n",
    "Of course! Let me help you with that. The query would be:\n\n",
    "Great question! To find this information, you can use:\n\n",
    "I'd be happy to help! Here's what you're looking for:\n\n",
    "Based on the schema, here is the appropriate query:\n\n",
]

FLUFF_SUFFIXES = [
    "\n\nThis query will return the results you're looking for. Let me know if you need anything else!",
    "\n\nHope this helps! Feel free to ask if you have more questions.",
    "\n\nThis should give you the answer. Would you like me to explain how it works?",
]


def _add_fluff(cypher: str) -> str:
    """Wrap valid Cypher in conversational fluff (rejected pattern)."""
    prefix = random.choice(FLUFF_PREFIXES)
    suffix = random.choice(FLUFF_SUFFIXES)
    return f"{prefix}```cypher\n{cypher}\n```{suffix}"


def _hallucinate_cypher(cypher: str) -> str:
    """Create a plausible but incorrect Cypher query (rejected pattern)."""
    mutations = [
        lambda c: re.sub(r"RETURN .+", "RETURN count(*) AS answer", c),
        lambda c: c.replace("MATCH", "OPTIONAL MATCH") + "\nRETURN 'unknown' AS answer",
        lambda c: re.sub(r"\{name: \$\w+\}", "{name: 'PLACEHOLDER'}", c),
        lambda c: re.sub(r"-\[:\w+\]->", "-[:RELATED_TO]->", c),
        lambda c: c + "\nUNION\nMATCH (n) RETURN n.name AS answer LIMIT 1",
    ]
    mutation = random.choice(mutations)
    return mutation(cypher)


def build_prompt(question: str, schema_summary: str) -> str:
    """Build the instruction prompt for Text2Cypher."""
    return (
        f"Given the following Neo4j schema:\n{schema_summary}\n\n"
        f"Generate a Cypher query to answer: {question}\n\n"
        f"Output only the Cypher query, nothing else."
    )


def generate_dpo_pairs(
    text2cypher_path: Path,
    output_path: Path,
    pairs_per_sample: int = 2,
) -> dict:
    """Generate DPO pairs from Text2Cypher training data.

    Input format (JSONL): {"question": ..., "schema": ..., "cypher": ...}
    Output format (JSONL): {"prompt": ..., "chosen": ..., "rejected": ...}
    """
    stats = {"total_input": 0, "pairs_generated": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(text2cypher_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total_input"] += 1
            sample = json.loads(line)

            question = sample["question"]
            schema = sample.get("schema", "")
            gold_cypher = sample["cypher"]

            prompt = build_prompt(question, schema)

            # Pair 1: chosen=clean cypher, rejected=fluff-wrapped cypher
            pair1 = {
                "prompt": prompt,
                "chosen": gold_cypher,
                "rejected": _add_fluff(gold_cypher),
            }
            fout.write(json.dumps(pair1, ensure_ascii=False) + "\n")
            stats["pairs_generated"] += 1

            # Pair 2: chosen=clean cypher, rejected=hallucinated cypher
            if pairs_per_sample >= 2:
                pair2 = {
                    "prompt": prompt,
                    "chosen": gold_cypher,
                    "rejected": _hallucinate_cypher(gold_cypher),
                }
                fout.write(json.dumps(pair2, ensure_ascii=False) + "\n")
                stats["pairs_generated"] += 1

    logger.info("DPO pairs generated", extra=stats)
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate DPO preference pairs")
    parser.add_argument("--input", type=Path, required=True, help="Text2Cypher training JSONL")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "dpo_pairs.jsonl")
    parser.add_argument("--pairs-per-sample", type=int, default=2)
    args = parser.parse_args()

    stats = generate_dpo_pairs(args.input, args.output, args.pairs_per_sample)
    print(f"Done: {stats['pairs_generated']} pairs from {stats['total_input']} samples")


if __name__ == "__main__":
    main()
