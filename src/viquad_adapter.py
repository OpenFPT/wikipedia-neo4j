"""Adapter for UIT-ViQuAD2.0 dataset from HuggingFace."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from src.logging_utils import get_logger

logger = get_logger(__name__)

VIQUAD_DATASET_ID = "taidng/UIT-ViQuAD2.0"
VIQUAD_CACHE_DIR = Path("data/viquad2")
VIQUAD_EVAL_PATH = Path("data/viquad2_eval.jsonl")


def _convert_sample(sample: dict) -> dict:
    """Convert a single HF sample to internal eval format."""
    answers = sample.get("answers", {})
    gold_texts = answers.get("text", []) if isinstance(answers, dict) else []

    return {
        "id": sample["id"],
        "question": sample["question"],
        "gold_answers": gold_texts,
        "context": sample["context"],
        "title": sample["title"],
        "is_impossible": sample.get("is_impossible", False),
    }


def load_viquad(split: str = "validation", limit: int | None = None) -> list[dict]:
    """Load ViQuAD2.0 and convert to internal eval format.

    Tries local JSONL cache first, falls back to HuggingFace download.
    """
    cache_path = VIQUAD_CACHE_DIR / f"{split}.jsonl"
    if cache_path.exists():
        logger.info(f"Loading ViQuAD2.0 from cache: {cache_path}")
        samples = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                samples.append(json.loads(line))
                if limit and len(samples) >= limit:
                    break
        return samples

    logger.info(f"Downloading ViQuAD2.0 split={split} from HuggingFace")
    ds = load_dataset(VIQUAD_DATASET_ID, split=split)

    samples = []
    for i, row in enumerate(ds):
        samples.append(_convert_sample(row))
        if limit and len(samples) >= limit:
            break

    logger.info(f"Loaded {len(samples)} samples from ViQuAD2.0 {split}")
    return samples


def export_eval_jsonl(split: str = "validation", output: Path | None = None) -> Path:
    """Export converted dataset to JSONL for offline use."""
    out_path = output or VIQUAD_CACHE_DIR / f"{split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading ViQuAD2.0 split={split} from HuggingFace")
    ds = load_dataset(VIQUAD_DATASET_ID, split=split)

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            record = _convert_sample(row)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Exported {count} samples to {out_path}")
    return out_path
