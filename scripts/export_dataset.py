"""Export Arrow dataset to JSONL/CSV for Neo4j ingestion."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_from_disk

from src.config import settings
from src.logging_utils import configure_logging, get_logger
from src.ner import extract_entities
from src.text_utils import chunk_text, normalize_vietnamese

configure_logging(settings.log_level, settings.json_logs, log_dir=settings.log_dir, task_name="export")
logger = get_logger(__name__)

_STOP = False


def _handle_sigint(sig, frame):
    global _STOP
    _STOP = True
    print("\nGraceful shutdown requested, finishing current batch...")


def _load_checkpoint(checkpoint_path: Path) -> int:
    if checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text())
        return data.get("last_index", 0)
    return 0


def _save_checkpoint(checkpoint_path: Path, last_index: int, stats: dict) -> None:
    tmp = checkpoint_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"last_index": last_index, **stats}, indent=2))
    tmp.rename(checkpoint_path)


def export_dataset(
    path: str,
    output_dir: str,
    limit: int | None,
    start: int,
    min_length: int,
    batch_size: int,
    skip_ner: bool,
    checkpoint_every: int,
) -> None:
    ds = load_from_disk(path)
    total = len(ds)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Export started", extra={"total": total, "min_length": min_length, "skip_ner": skip_ner})

    checkpoint_path = out / ".checkpoint.json"
    resumed_from = _load_checkpoint(checkpoint_path)
    if start == 0 and resumed_from > 0:
        start = resumed_from
        print(f"Resuming from checkpoint at index {start}")

    end = total if limit is None else min(total, start + limit)

    pages_file = out / "pages.jsonl"
    chunks_file = out / "chunks.jsonl"
    entities_file = out / "entities.jsonl"
    mentions_file = out / "mentions.csv"

    mode = "a" if start > 0 else "w"
    fp_pages = open(pages_file, mode, encoding="utf-8")
    fp_chunks = open(chunks_file, mode, encoding="utf-8")
    fp_entities = open(entities_file, mode, encoding="utf-8")
    fp_mentions = open(mentions_file, mode, encoding="utf-8")

    if mode == "w":
        fp_mentions.write("chunk_id,entity_id,relation_type\n")

    stats = {"pages": 0, "chunks": 0, "entities": 0, "mentions": 0, "skipped": 0}
    seen_entities: set[str] = set()
    last_idx = start

    try:
        for idx in range(start, end):
            if _STOP:
                print(f"Stopped at index {idx}")
                break
            last_idx = idx

            row = ds[idx]
            text = str(row.get("text", "")).strip()
            if len(text) < min_length:
                stats["skipped"] += 1
                continue

            page_id = str(row.get("id", ""))
            title = str(row.get("title", "")).strip() or f"untitled-{uuid.uuid4()}"
            url = str(row.get("url", "")).strip()
            if not page_id:
                page_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url or title))

            text = normalize_vietnamese(text)
            summary = text[:400]

            fp_pages.write(json.dumps(
                {"id": page_id, "title": title, "url": url, "summary": summary},
                ensure_ascii=False,
            ) + "\n")
            stats["pages"] += 1

            chunks = chunk_text(text)
            chunk_ids = []
            for seq, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{page_id}#chunk#{seq}"))
                chunk_ids.append(chunk_id)
                fp_chunks.write(json.dumps(
                    {"chunk_id": chunk_id, "page_id": page_id, "text": chunk, "seq": seq},
                    ensure_ascii=False,
                ) + "\n")
                stats["chunks"] += 1

            if not skip_ner:
                entities = extract_entities(text)
                for name, entity_type in entities:
                    entity_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name.lower()))
                    if entity_id not in seen_entities:
                        seen_entities.add(entity_id)
                        label = entity_type if entity_type in (
                            "Person", "Organization", "Location", "Work", "Event"
                        ) else "Entity"
                        fp_entities.write(json.dumps(
                            {"entity_id": entity_id, "name": name, "label": label, "aliases": ""},
                            ensure_ascii=False,
                        ) + "\n")
                        stats["entities"] += 1

                    for chunk_id in chunk_ids:
                        rel_type = f"MENTIONS_{entity_type.upper()}" if entity_type != "Entity" else "MENTIONS"
                        fp_mentions.write(f"{chunk_id},{entity_id},{rel_type}\n")
                        stats["mentions"] += 1

            if stats["pages"] % batch_size == 0:
                pct = (idx - start) / max(end - start, 1) * 100
                print(f"  [{pct:5.1f}%] {stats['pages']:,} pages, {stats['chunks']:,} chunks, {stats['entities']:,} entities")

            if stats["pages"] % checkpoint_every == 0:
                _save_checkpoint(checkpoint_path, idx + 1, stats)

    finally:
        fp_pages.close()
        fp_chunks.close()
        fp_entities.close()
        fp_mentions.close()
        _save_checkpoint(checkpoint_path, last_idx + 1, stats)

    stats_file = out / "stats.json"
    stats_file.write_text(json.dumps(stats, indent=2))
    logger.info("Export complete", extra=stats)
    print(f"\nExport complete: {stats}")


def main() -> None:
    signal.signal(signal.SIGINT, _handle_sigint)

    parser = argparse.ArgumentParser(description="Export ViWiki dataset to JSONL/CSV")
    parser.add_argument("--path", default="data/viwiki-cleaned", help="Path to Arrow dataset")
    parser.add_argument("--output-dir", default="data/export", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max articles to process")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--min-length", type=int, default=settings.min_text_length, help="Min text length filter")
    parser.add_argument("--batch-size", type=int, default=settings.ingest_batch_size, help="Log progress every N")
    parser.add_argument("--skip-ner", action="store_true", help="Skip NER extraction")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Save checkpoint every N pages")
    args = parser.parse_args()

    export_dataset(
        path=args.path,
        output_dir=args.output_dir,
        limit=args.limit,
        start=args.start,
        min_length=args.min_length,
        batch_size=args.batch_size,
        skip_ner=args.skip_ner,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
