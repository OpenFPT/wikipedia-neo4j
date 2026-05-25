"""QLoRA fine-tuning script for Text2Cypher using Unsloth + TRL.

Two-stage training:
  Stage 1 (SFT): Supervised fine-tuning for executable Cypher generation.
  Stage 2 (GRPO): Group Relative Policy Optimization with execution reward (placeholder).

Usage:
    uv run python scripts/finetune_text2cypher.py --data data/training/text2cypher_train.jsonl
    uv run python scripts/finetune_text2cypher.py --dry-run  # validate setup only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_utils import configure_logging, get_logger

configure_logging("INFO", log_dir="logs", task_name="finetune")
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "AITeamVN/Vi-Qwen2-7B-RAG"
DEFAULT_OUTPUT_DIR = "models/text2cypher-lora"

# QLoRA hyperparameters
QLORA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# SFT prompt template
SFT_PROMPT_TEMPLATE = """Given the following graph schema and question, generate a Cypher query.

Schema: {schema}

Question: {question}

Cypher: {cypher}"""

SFT_INFERENCE_TEMPLATE = """Given the following graph schema and question, generate a Cypher query.

Schema: {schema}

Question: {question}

Cypher: """


def setup_model_and_tokenizer(model_id: str):
    """Load base model with QLoRA config via Unsloth for efficient training.

    Returns:
        Tuple of (model, tokenizer) with LoRA adapters applied.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error(
            "unsloth not installed. Install training dependencies: "
            "uv sync --group training"
        )
        sys.exit(1)

    logger.info("Loading model with Unsloth", extra={"model": model_id})

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=QLORA_CONFIG["r"],
        lora_alpha=QLORA_CONFIG["lora_alpha"],
        lora_dropout=QLORA_CONFIG["lora_dropout"],
        target_modules=QLORA_CONFIG["target_modules"],
        bias=QLORA_CONFIG["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    logger.info("Model loaded with LoRA adapters")
    return model, tokenizer


def prepare_dataset(data_path: str, tokenizer, max_samples: int | None = None):
    """Load and format training data for SFT.

    Args:
        data_path: Path to JSONL with {question, cypher, schema} entries.
        tokenizer: Tokenizer for formatting.
        max_samples: Optional limit on number of samples.

    Returns:
        HuggingFace Dataset ready for SFT training.
    """
    from datasets import Dataset

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text = SFT_PROMPT_TEMPLATE.format(
                schema=entry["schema"],
                question=entry["question"],
                cypher=entry["cypher"],
            )
            samples.append({"text": text})
            if max_samples and len(samples) >= max_samples:
                break

    logger.info("Dataset loaded", extra={"samples": len(samples)})
    dataset = Dataset.from_list(samples)
    return dataset


def train_sft(
    model,
    tokenizer,
    dataset,
    output_dir: str,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
    max_seq_length: int = 2048,
):
    """Stage 1: Supervised fine-tuning for executable Cypher.

    Args:
        model: Model with LoRA adapters.
        tokenizer: Tokenizer.
        dataset: Formatted training dataset.
        output_dir: Directory to save LoRA adapter.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Per-device batch size.
        max_seq_length: Maximum sequence length.
    """
    from trl import SFTTrainer
    from transformers import TrainingArguments

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        fp16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=True,
    )

    logger.info(
        "Starting SFT training",
        extra={"epochs": epochs, "lr": lr, "batch_size": batch_size},
    )
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("LoRA adapter saved", extra={"path": output_dir})


def train_grpo(model, tokenizer, dataset, output_dir: str):
    """Stage 2: GRPO with execution reward.

    This stage uses Group Relative Policy Optimization to further refine
    the model using execution-based rewards (does the generated Cypher
    actually run and return correct results?).

    NOTE: This is a placeholder. Full GRPO implementation requires:
    - A Neo4j sandbox for safe query execution
    - Reward function comparing execution results to gold answers
    - TRL's GRPOTrainer configuration
    """
    logger.info(
        "GRPO training stage not yet implemented. "
        "Stage 1 SFT adapter is saved and ready for use."
    )
    # Future implementation:
    # from trl import GRPOTrainer, GRPOConfig
    # def reward_fn(queries, responses):
    #     """Execute Cypher and score based on correctness."""
    #     ...
    # grpo_config = GRPOConfig(...)
    # trainer = GRPOTrainer(model=model, config=grpo_config, reward_fn=reward_fn, ...)
    # trainer.train()


def validate_dry_run(model_id: str, data_path: str):
    """Validate setup without full training: load model, check 10 samples, run 1 step."""
    logger.info("Dry run: validating setup...")

    # Check data file
    data_file = Path(data_path)
    if not data_file.exists():
        logger.error("Training data not found", extra={"path": data_path})
        sys.exit(1)

    # Count samples
    sample_count = 0
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample_count += 1
    logger.info("Training data validated", extra={"samples": sample_count})

    # Try loading model
    logger.info("Loading model for validation...")
    model, tokenizer = setup_model_and_tokenizer(model_id)

    # Load small dataset
    dataset = prepare_dataset(data_path, tokenizer, max_samples=10)
    logger.info("Dry run dataset loaded", extra={"samples": len(dataset)})

    # Run 1 training step
    from trl import SFTTrainer
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir="/tmp/text2cypher_dry_run",
        num_train_epochs=1,
        max_steps=1,
        per_device_train_batch_size=1,
        logging_steps=1,
        fp16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=512,
    )
    trainer.train()

    logger.info("Dry run complete: setup is valid")
    print("Dry run passed. Model loads correctly and training step executes.")


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Text2Cypher (Vi-Qwen2-7B-RAG)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/training/text2cypher_train.jsonl",
        help="Path to training JSONL (default: data/training/text2cypher_train.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for LoRA adapter (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Base model HuggingFace ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without full training",
    )
    parser.add_argument(
        "--grpo",
        action="store_true",
        help="Run GRPO stage after SFT (not yet implemented)",
    )
    args = parser.parse_args()

    if args.dry_run:
        validate_dry_run(args.model_id, args.data)
        return

    # Validate data exists
    data_file = Path(args.data)
    if not data_file.exists():
        logger.error("Training data not found", extra={"path": args.data})
        print(
            f"Error: {args.data} not found. "
            "Run scripts/generate_training_data.py first."
        )
        sys.exit(1)

    # Stage 1: SFT
    model, tokenizer = setup_model_and_tokenizer(args.model_id)
    dataset = prepare_dataset(args.data, tokenizer)
    train_sft(
        model,
        tokenizer,
        dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    # Stage 2: GRPO (optional)
    if args.grpo:
        train_grpo(model, tokenizer, dataset, args.output_dir)

    print(f"Training complete. LoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
