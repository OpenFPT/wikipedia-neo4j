"""Upload ViWiki-MHR dataset to HuggingFace Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value, Sequence
from huggingface_hub import HfApi

from src.logging_utils import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "viwiki_mhr" / "final"

FEATURES = Features({
    "id": Value("string"),
    "question": Value("string"),
    "answer": Value("string"),
    "reasoning_type": Value("string"),
    "is_answerable": Value("bool"),
    "gold_passage_ids": Sequence(Value("string")),
    "cypher_query": Value("string"),
    "decomposition_annotations": {
        "sub_questions": Sequence(Value("string")),
        "sub_answers": Sequence(Value("string")),
        "reasoning_chain": Sequence(Value("string")),
    },
    "source": Value("string"),
    "num_hops": Value("int32"),
})


def load_split(split_path: Path) -> Dataset:
    """Load a JSONL split into a HuggingFace Dataset."""
    return Dataset.from_json(str(split_path))


def upload_dataset(
    repo_id: str,
    data_dir: Path = DATA_DIR,
    private: bool = False,
) -> str:
    """Upload train/dev/test splits to HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo (e.g., "username/viwiki-mhr")
        data_dir: Directory containing train.jsonl, dev.jsonl, test.jsonl
        private: Whether to make the dataset private

    Returns:
        URL of the uploaded dataset.
    """
    splits = {}
    for split_name in ["train", "dev", "test"]:
        split_file = data_dir / f"{split_name}.jsonl"
        if split_file.exists():
            splits[split_name] = load_split(split_file)
            logger.info(f"Loaded {split_name}", extra={"rows": len(splits[split_name])})
        else:
            logger.warning(f"Split file not found: {split_file}")

    if not splits:
        raise FileNotFoundError(f"No split files found in {data_dir}")

    dataset_dict = DatasetDict(splits)

    dataset_dict.push_to_hub(
        repo_id,
        private=private,
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Dataset uploaded", extra={"url": url})
    return url


def create_dataset_card(repo_id: str) -> str:
    """Generate a dataset card (README.md) for HuggingFace."""
    return f"""---
language:
  - vi
license: cc-by-sa-4.0
task_categories:
  - question-answering
  - text-generation
tags:
  - multi-hop-reasoning
  - knowledge-graph
  - vietnamese
  - wikipedia
size_categories:
  - 10K<n<100K
---

# ViWiki-MHR: Vietnamese Wikipedia Multi-Hop Reasoning Dataset

## Dataset Description

ViWiki-MHR is a Vietnamese multi-hop question answering dataset built from Vietnamese Wikipedia,
designed to evaluate and train models on complex reasoning over knowledge graphs.

### Dataset Summary

- **Total samples:** ~36K
- **Single-hop (UIT-ViQuAD reformatted):** ~28K
- **Multi-hop (synthetic from KG walks):** ~7K
- **Adversarial unanswerable:** ~1K
- **Language:** Vietnamese
- **Source:** Vietnamese Wikipedia (20231101 dump)

### Reasoning Types

| Type | Description | Count |
|------|-------------|-------|
| bridge_2hop | A→B→C chain reasoning | ~3K |
| bridge_3hop | A→B→C→D chain reasoning | ~1.5K |
| comparison | Two entities sharing a property | ~1K |
| intersection | Two paths converging | ~1K |
| temporal | Time-constrained reasoning | ~500 |

### Data Fields

- `id`: Unique sample identifier
- `question`: Vietnamese question text
- `answer`: Gold answer text
- `reasoning_type`: Type of reasoning required
- `is_answerable`: Whether the question is answerable from the corpus
- `gold_passage_ids`: List of paragraph IDs containing evidence
- `cypher_query`: Gold Cypher query to retrieve the answer from Neo4j
- `decomposition_annotations`: Sub-questions and sub-answers breakdown
- `source`: Origin of the sample (uit_viquad, synthetic, adversarial)
- `num_hops`: Number of reasoning hops required

### Quality Control

All samples pass a 3-stage QC pipeline:
1. **Grounding filter:** Answer must appear in gold passages
2. **NLI entailment:** Cross-encoder verifies answer entailment
3. **Well-formedness:** LLM verifier checks question quality

Test split includes 1500+ human-verified samples.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
train = dataset["train"]
print(train[0])
```

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{viwiki_mhr_2024,
  title={{ViWiki-MHR: Vietnamese Wikipedia Multi-Hop Reasoning Dataset}},
  year={{2024}},
  publisher={{HuggingFace}},
}}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload ViWiki-MHR to HuggingFace")
    parser.add_argument("--repo-id", type=str, required=True, help="HF repo ID (e.g., user/viwiki-mhr)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--card-only", action="store_true", help="Only print dataset card")
    args = parser.parse_args()

    if args.card_only:
        print(create_dataset_card(args.repo_id))
        return

    url = upload_dataset(args.repo_id, args.data_dir, args.private)
    print(f"Uploaded: {url}")


if __name__ == "__main__":
    main()
