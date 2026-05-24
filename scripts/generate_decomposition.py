"""Generate decomposition annotations: sub-questions + sub-answers for multi-hop QA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Decomposition:
    """A decomposed multi-hop question."""

    sub_questions: list[str]
    sub_answers: list[str]
    reasoning_chain: list[str]


DECOMPOSITION_PATTERNS: dict[str, list[str]] = {
    "bridge_2hop": [
        "What is the {rel1} of {entity1}?",
        "What is the {rel2} of [answer1]?",
    ],
    "bridge_3hop": [
        "What is the {rel1} of {entity1}?",
        "What is the {rel2} of [answer1]?",
        "What is the {rel3} of [answer2]?",
    ],
    "comparison": [
        "What is the {rel1} of {entity1}?",
        "What is the {rel2} of {entity2}?",
        "What do [answer1] and [answer2] have in common?",
    ],
    "intersection": [
        "What entities are related to {entity1} via {rel1}?",
        "What entities are related to {entity2} via {rel2}?",
        "Which entity appears in both [answer1] and [answer2]?",
    ],
    "temporal": [
        "What events involve {entity1}?",
        "Which of those events occurred {temporal_constraint}?",
    ],
}


def decompose_sample(sample: dict) -> Decomposition | None:
    """Generate decomposition for a single multi-hop QA sample.

    Args:
        sample: dict with keys:
            - reasoning_type: bridge_2hop, bridge_3hop, comparison, intersection, temporal
            - kg_path: list of hop info
            - entities: dict of entity names
            - intermediate_answers: list of intermediate answers (if available)
            - answer: final answer

    Returns:
        Decomposition object or None if pattern not recognized.
    """
    reasoning_type = sample.get("reasoning_type", "")
    entities = sample.get("entities", {})
    kg_path = sample.get("kg_path", [])
    intermediate_answers = sample.get("intermediate_answers", [])
    final_answer = sample.get("answer", "")

    patterns = DECOMPOSITION_PATTERNS.get(reasoning_type)
    if patterns is None:
        return None

    sub_questions = []
    for pattern in patterns:
        q = pattern
        for key, value in entities.items():
            q = q.replace(f"{{{key}}}", value)
        for i, hop in enumerate(kg_path):
            q = q.replace(f"{{rel{i+1}}}", hop.get("relationship_nl", hop.get("relationship", "")))
        if "temporal_constraint" in q:
            q = q.replace("{temporal_constraint}", sample.get("temporal_constraint", ""))
        sub_questions.append(q)

    # Build sub-answers from intermediate_answers + final
    sub_answers = list(intermediate_answers)
    if final_answer and (not sub_answers or sub_answers[-1] != final_answer):
        sub_answers.append(final_answer)

    # Pad sub_answers to match sub_questions length
    while len(sub_answers) < len(sub_questions):
        sub_answers.append("")

    # Build reasoning chain
    reasoning_chain = []
    for i, (q, a) in enumerate(zip(sub_questions, sub_answers)):
        step = f"Step {i+1}: {q} → {a}" if a else f"Step {i+1}: {q} → [unknown]"
        reasoning_chain.append(step)

    return Decomposition(
        sub_questions=sub_questions,
        sub_answers=sub_answers[:len(sub_questions)],
        reasoning_chain=reasoning_chain,
    )


def process_dataset(input_path: Path, output_path: Path) -> dict:
    """Add decomposition_annotations field to each sample."""
    stats = {"total": 0, "decomposed": 0, "skipped": 0}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            sample = json.loads(line)

            decomp = decompose_sample(sample)
            if decomp:
                sample["decomposition_annotations"] = {
                    "sub_questions": decomp.sub_questions,
                    "sub_answers": decomp.sub_answers,
                    "reasoning_chain": decomp.reasoning_chain,
                }
                stats["decomposed"] += 1
            else:
                sample["decomposition_annotations"] = None
                stats["skipped"] += 1

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Decomposition complete", extra=stats)
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate decomposition annotations")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stats = process_dataset(args.input, args.output)
    print(f"Done: {stats['decomposed']}/{stats['total']} decomposed, {stats['skipped']} skipped")


if __name__ == "__main__":
    main()
