from __future__ import annotations

import argparse
import uuid

from datasets import load_from_disk

from src.ingestion.pipeline import _upsert_page_from_text
from src.logging_utils import get_logger


logger = get_logger(__name__)


def ingest_local_dataset(path: str, limit: int | None, start: int, log_every: int) -> int:
    ds = load_from_disk(path)
    total = len(ds)
    end = total if limit is None else min(total, start + limit)
    processed = 0

    for idx in range(start, end):
        row = ds[idx]
        page_id = str(row.get("id", ""))
        title = str(row.get("title", "")).strip() or f"untitled-{uuid.uuid4()}"
        url = str(row.get("url", "")).strip() or f"https://example.org/{title.replace(' ', '_')}"
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        summary = text[:400]
        if not page_id:
            page_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url))

        _upsert_page_from_text(
            page_id=page_id,
            title=title,
            url=url,
            text=text,
            summary=summary,
        )
        processed += 1

        if processed % log_every == 0:
            logger.info("Local ingest progress", extra={"processed": processed, "last_title": title})

    logger.info(
        "Local ingest completed",
        extra={"processed": processed, "start": start, "end": end, "dataset_total": total},
    )
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local HF dataset saved via save_to_disk")
    parser.add_argument("--path", default="data/viet-wikipedia", help="Path to dataset folder")
    parser.add_argument("--limit", type=int, default=5, help="Max records to ingest (default: 5)")
    parser.add_argument("--start", type=int, default=0, help="Start index in dataset")
    parser.add_argument("--log-every", type=int, default=10, help="Log progress every N records")
    args = parser.parse_args()

    ingest_local_dataset(args.path, args.limit, args.start, args.log_every)


if __name__ == "__main__":
    main()
