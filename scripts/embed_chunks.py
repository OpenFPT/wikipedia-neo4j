"""Generate embeddings for exported chunks and write to JSONL."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.llm import embed_texts_batch
from src.logging_utils import configure_logging, get_logger

configure_logging(settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="embed")
logger = get_logger(__name__)

_STOP = False


def _handle_sigint(sig, frame):
    global _STOP
    _STOP = True
    print("\nGraceful shutdown requested, finishing current batch...")


def _load_checkpoint(checkpoint_path: Path) -> int:
    if checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text())
        return data.get("last_line", 0)
    return 0


def _save_checkpoint(checkpoint_path: Path, last_line: int) -> None:
    tmp = checkpoint_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"last_line": last_line}))
    tmp.rename(checkpoint_path)


def embed_chunks(
    input_path: str,
    output_path: str,
    limit: int | None,
    batch_size: int,
    checkpoint_every: int,
) -> None:
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out.with_suffix(".checkpoint.json")
    start_line = _load_checkpoint(checkpoint_path)
    if start_line > 0:
        print(f"Resuming from line {start_line}")

    mode = "a" if start_line > 0 else "w"
    fp_out = open(out, mode, encoding="utf-8")

    processed = 0
    batch_texts: list[str] = []
    batch_ids: list[str] = []

    try:
        with open(inp, "r", encoding="utf-8") as fp_in:
            for line_num, line in enumerate(fp_in):
                if line_num < start_line:
                    continue
                if _STOP:
                    print(f"Stopped at line {line_num}")
                    break
                if limit and processed >= limit:
                    break

                row = json.loads(line)
                batch_texts.append(row["text"])
                batch_ids.append(row["chunk_id"])

                if len(batch_texts) >= batch_size:
                    embeddings = embed_texts_batch(batch_texts, batch_size=batch_size)
                    for chunk_id, embedding in zip(batch_ids, embeddings):
                        fp_out.write(json.dumps(
                            {"chunk_id": chunk_id, "embedding": embedding},
                        ) + "\n")
                    processed += len(batch_texts)
                    batch_texts.clear()
                    batch_ids.clear()

                    if processed % checkpoint_every == 0:
                        fp_out.flush()
                        _save_checkpoint(checkpoint_path, line_num + 1)
                        print(f"  Embedded {processed:,} chunks")

            if batch_texts and not _STOP:
                embeddings = embed_texts_batch(batch_texts, batch_size=batch_size)
                for chunk_id, embedding in zip(batch_ids, embeddings):
                    fp_out.write(json.dumps(
                        {"chunk_id": chunk_id, "embedding": embedding},
                    ) + "\n")
                processed += len(batch_texts)

    finally:
        fp_out.close()
        _save_checkpoint(checkpoint_path, start_line + processed)

    print(f"\nEmbedding complete: {processed:,} chunks embedded → {out}")


def main() -> None:
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(description="Generate embeddings for exported chunks")
    parser.add_argument("--input", default="data/export/chunks.jsonl", help="Input chunks JSONL")
    parser.add_argument("--output", default="data/export/chunk_embeddings.jsonl", help="Output embeddings JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to embed")
    parser.add_argument("--batch-size", type=int, default=settings.embed_batch_size, help="Embedding batch size")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Checkpoint every N chunks")
    parser.add_argument("--backend", choices=["local", "gemini"], default=settings.embedding_backend, help="Embedding backend")
    args = parser.parse_args()

    if args.backend != settings.embedding_backend:
        settings.embedding_backend = args.backend

    embed_chunks(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
