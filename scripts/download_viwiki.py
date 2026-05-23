"""Download Vietnamese Wikipedia dump from HuggingFace and convert to paragraphs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "viwiki_paragraphs.parquet"
HF_CONFIG = "20231101.vi"
MIN_PARAGRAPH_LENGTH = 50


def split_into_paragraphs(text: str) -> list[str]:
    """Split article text into meaningful paragraphs."""
    paragraphs = re.split(r"\n{2,}", text)
    result = []
    for p in paragraphs:
        cleaned = p.strip()
        if len(cleaned) >= MIN_PARAGRAPH_LENGTH:
            result.append(cleaned)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download Vietnamese Wikipedia dump")
    parser.add_argument("--max-articles", type=int, default=None, help="Limit number of articles")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading wikimedia/wikipedia ({HF_CONFIG}) from HuggingFace (streaming)...")
    ds = load_dataset("wikimedia/wikipedia", HF_CONFIG, split="train", streaming=True)

    rows: list[dict] = []
    article_count = 0

    for article in ds:
        if args.max_articles and article_count >= args.max_articles:
            break

        article_id = str(article.get("id", ""))
        title = str(article.get("title", "")).strip()
        url = str(article.get("url", "")).strip()
        text = str(article.get("text", "")).strip()

        if not text or not title:
            continue

        paragraphs = split_into_paragraphs(text)
        for idx, para in enumerate(paragraphs):
            para_id = hashlib.md5(f"{article_id}:{idx}".encode()).hexdigest()
            rows.append({
                "paragraph_id": para_id,
                "article_id": article_id,
                "title": title,
                "url": url,
                "paragraph_index": idx,
                "text": para,
            })

        article_count += 1
        if article_count % 5000 == 0:
            print(f"  Processed {article_count} articles, {len(rows)} paragraphs so far...")

    print(f"Total: {article_count} articles, {len(rows)} paragraphs")
    print(f"Writing to {OUTPUT_FILE}...")

    table = pa.table({
        "paragraph_id": pa.array([r["paragraph_id"] for r in rows], type=pa.string()),
        "article_id": pa.array([r["article_id"] for r in rows], type=pa.string()),
        "title": pa.array([r["title"] for r in rows], type=pa.string()),
        "url": pa.array([r["url"] for r in rows], type=pa.string()),
        "paragraph_index": pa.array([r["paragraph_index"] for r in rows], type=pa.int32()),
        "text": pa.array([r["text"] for r in rows], type=pa.string()),
    })
    pq.write_table(table, OUTPUT_FILE, compression="snappy")
    print(f"Done. Output: {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
