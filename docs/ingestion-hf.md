# Ingestion Mode: Hugging Face and Local Wikipedia Datasets

Supported dataset sources:

- `wikimedia/wikipedia` for the upstream Hugging Face Wikipedia loader.
- `Keithsel/viwiki-20260523` for the project-published Vietnamese Wikipedia snapshot produced from raw MediaWiki XML.

Common config format: `<dump>.<lang>` (for example `20231101.vi`, `20231101.en`, `20231101.simple`).

Default config targets Vietnamese Wikipedia (`20231101.vi`).

For reproducible thesis runs, prefer the pinned project dataset:

```python
from datasets import load_dataset

dataset = load_dataset("Keithsel/viwiki-20260523")
```

## Synchronous endpoint

`POST /ingest/hf`

```bash
curl -X POST "http://localhost:8000/ingest/hf" \
  -H "Content-Type: application/json" \
  -d '{"config_name":"20231101.vi","split":"train","sample_size":2,"streaming":true}'
```

## Streaming mode

Use `"streaming": true` for large language subsets to avoid loading full dataset into memory.

## Recommended for large imports

Use background jobs (`/ingest/hf/jobs`) so API remains responsive during ingestion.

See [Background Jobs](api/background-jobs.md) for async job management.

## Raw XML dump processing

The repository includes a raw Vietnamese Wikipedia dump conversion utility in
`scripts/viwiki_processing/`. It streams MediaWiki XML with `lxml`, keeps only
main-namespace articles (`ns=0`), cleans wikitext with `mwparserfromhell`, and
writes two Parquet dataset folders:

- `articles_cleaned/` — cleaned plain text for ingestion and retrieval.
- `articles_raw/` — original wikitext for link/template graph extraction.

Example conversion:

```bash
uv sync --group xml-processing
python -m scripts.viwiki_processing.cli convert \
  --xml dumps/viwiki-20260523-pages-articles.xml \
  --output articles_cleaned \
  --raw-output articles_raw
```

If `--xml` is omitted, the converter processes every `*.xml` file under `./dumps`.
Use `--batch-size` to control Parquet shard size and `--limit` for smoke tests.

The published HF dataset `Keithsel/viwiki-20260523` is the shareable output of
this raw XML processing path and should be referenced by reports and downstream
experiments instead of transient local dump folders.

## Pipeline

HF and locally converted Parquet ingestion follow the same graph ingestion stages as Wikipedia API ingestion:

1. Extract text from HF dataset records
2. Chunk text
3. NER extraction (configurable backend)
4. Embedding generation
5. Write to Neo4j with typed entities and relationships
