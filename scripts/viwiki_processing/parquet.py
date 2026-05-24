import os
import sys
import time
import urllib.parse
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from scripts.viwiki_processing.wikitext import clean_wikitext
from scripts.viwiki_processing.xml_dump import parse_wiki_dump


PARQUET_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("url", pa.string()),
        ("title", pa.string()),
        ("text", pa.string()),
    ]
)
DEFAULT_CLEAN_OUTPUT_DIR = "articles_cleaned"
DEFAULT_RAW_OUTPUT_DIR = "articles_raw"
DEFAULT_SHARD_SIZE = 100000
DEFAULT_ROW_GROUP_SIZE = 1000


def article_url(title):
    encoded_title = urllib.parse.quote(title)
    return f"https://vi.wikipedia.org/wiki/{encoded_title}"


def empty_batch():
    return {
        "id": [],
        "url": [],
        "title": [],
        "text": [],
    }


def convert_xml_to_parquet(
    xml_path,
    parquet_path=DEFAULT_CLEAN_OUTPUT_DIR,
    batch_size=DEFAULT_SHARD_SIZE,
    limit=None,
    raw_parquet_path=DEFAULT_RAW_OUTPUT_DIR,
):
    """
    Converts Wikipedia XML dump into a Parquet file matching the reference schema.
    Uses incremental streaming to scale to large files.

    By default this writes two Parquet dataset folders:
    - articles_cleaned/ with cleaned plain text
    - articles_raw/ with original raw article wikitext

    File paths ending in .parquet are still supported. Other paths are treated
    as dataset folders and receive sharded part-xxxxx.parquet files.
    """
    return convert_xml_dumps_to_parquet(
        xml_paths=[xml_path],
        parquet_path=parquet_path,
        batch_size=batch_size,
        limit=limit,
        raw_parquet_path=raw_parquet_path,
    )


def convert_xml_dumps_to_parquet(
    xml_paths,
    parquet_path=DEFAULT_CLEAN_OUTPUT_DIR,
    batch_size=DEFAULT_SHARD_SIZE,
    limit=None,
    raw_parquet_path=DEFAULT_RAW_OUTPUT_DIR,
):
    clean_output = ParquetOutput(parquet_path)
    raw_output = ParquetOutput(raw_parquet_path) if raw_parquet_path else None

    print(f"Writing cleaned articles to: {clean_output.display_path}")
    if raw_output:
        print(f"Writing raw articles to: {raw_output.display_path}")

    batch = empty_batch()
    raw_batch = empty_batch() if raw_output else None
    count = 0
    start_time = time.time()

    try:
        for xml_path in xml_paths:
            for page in parse_wiki_dump(xml_path, limit=_remaining_limit(limit, count)):
                title = page["title"]
                raw_text = page["text"]
                url = article_url(title)

                batch["id"].append(str(page["id"]))
                batch["url"].append(url)
                batch["title"].append(title)
                batch["text"].append(clean_wikitext(raw_text))

                if raw_batch is not None:
                    raw_batch["id"].append(str(page["id"]))
                    raw_batch["url"].append(url)
                    raw_batch["title"].append(title)
                    raw_batch["text"].append(raw_text)

                count += 1

                if len(batch["id"]) >= batch_size:
                    write_batch(clean_output, batch)
                    if raw_output and raw_batch is not None:
                        write_batch(raw_output, raw_batch)
                    print(
                        f"Wrote shard of {len(batch['id'])} articles to Parquet. (Total written: {count})"
                    )
                    batch = empty_batch()
                    raw_batch = empty_batch() if raw_output else None

                if limit and count >= limit:
                    break

            if limit and count >= limit:
                break

        if batch["id"]:
            write_batch(clean_output, batch)
            if raw_output and raw_batch is not None:
                write_batch(raw_output, raw_batch)
            print(
                f"Wrote final shard of {len(batch['id'])} articles to Parquet. (Total written: {count})"
            )
    finally:
        clean_output.close()
        if raw_output:
            raw_output.close()

    elapsed = time.time() - start_time
    print(f"Successfully converted XML to Parquet in {elapsed:.2f}s! Total articles: {count}")
    return count


def _remaining_limit(limit, count):
    if limit is None:
        return None
    return max(limit - count, 0)


def write_batch(output, batch):
    table = pa.Table.from_pydict(batch, schema=PARQUET_SCHEMA)
    output.write_table(table)


class ParquetOutput:
    def __init__(self, output_path):
        self.path = Path(output_path)
        self.writer = None
        self.part_index = 0
        self.is_single_file = self.path.suffix == ".parquet"

        if self.is_single_file:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(str(self.path), PARQUET_SCHEMA, compression="snappy")
            self.display_path = str(self.path)
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            for existing_part in self.path.glob("part-*.parquet"):
                existing_part.unlink()
            self.display_path = str(self.path)

    def write_table(self, table):
        if self.writer:
            self.writer.write_table(table, row_group_size=DEFAULT_ROW_GROUP_SIZE)
            return

        part_path = self.path / f"part-{self.part_index:05d}.parquet"
        pq.write_table(
            table,
            part_path,
            compression="snappy",
            row_group_size=DEFAULT_ROW_GROUP_SIZE,
        )
        self.part_index += 1

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None


def query_article_from_parquet(parquet_path, target_title):
    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file '{parquet_path}' not found.", file=sys.stderr)
        return None

    print(f"Searching for '{target_title}' in {parquet_path} using pyarrow dataset filtering...")
    start_time = time.time()

    try:
        dataset = ds.dataset(parquet_path, format="parquet")
        filter_expr = ds.field("title") == target_title
        table = dataset.to_table(filter=filter_expr)

        if len(table) == 0:
            print("Exact match not found. Trying case-insensitive search...")
            projection = {"title": ds.field("title"), "id": ds.field("id")}
            scanner = dataset.scanner(columns=projection)

            target_lower = target_title.lower().strip()
            matched_title = None

            for batch in scanner.to_batches():
                titles = batch.column("title").to_pylist()
                for title in titles:
                    if title.lower().strip() == target_lower:
                        matched_title = title
                        break
                if matched_title:
                    break

            if matched_title:
                filter_expr = ds.field("title") == matched_title
                table = dataset.to_table(filter=filter_expr)

        if len(table) == 0:
            elapsed = time.time() - start_time
            print(f"No match found. (Took {elapsed:.2f}s)")
            return None

        row = table.to_pylist()[0]
        elapsed = time.time() - start_time
        print(f"Found match: '{row['title']}' (ID: {row['id']}) in {elapsed:.4f}s!")
        return row
    except Exception as e:
        print(f"Error querying parquet: {e}", file=sys.stderr)
        return None
