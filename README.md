# Wikipedia Neo4j GraphRAG Demo

[![CI](https://github.com/OpenFPT/wikipedia-neo4j/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenFPT/wikipedia-neo4j/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.12%2B-blue) ![Coverage](https://img.shields.io/badge/coverage-75%25%2B-brightgreen) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GraphRAG system that ingests Vietnamese Wikipedia content into a Neo4j knowledge graph, answers multi-hop questions using graph-based retrieval, and generates QA datasets.

## Modes

- **Ingest mode**: build graph context from Wikipedia topics, raw XML-derived Parquet, or HF dataset (`/ingest`, `/ingest/hf`, async jobs).
- **Query mode (API)**: Gemini generates validated read-only Cypher with hybrid fulltext fallback (`/query`).
- **Query mode (Local)**: ReAct agent loop with graph tools using Qwen2.5-7B-Instruct (`/query`).
- **Dataset generation**: extract KG walks and produce multi-hop QA pairs (ViWiki-MHR).
- **Ops mode**: health/readiness/metrics/logging for deployment safety.

## Features

- Ingestion sources:
  - Wikipedia API (`POST /ingest`)
  - Hugging Face dataset (`POST /ingest/hf`)
  - Async HF jobs (`POST /ingest/hf/jobs`)
  - Raw Vietnamese Wikipedia XML conversion (`scripts/viwiki_processing/`) to cleaned/raw Parquet
- Pluggable NER: `simple` (regex), `underthesea`, or `phonlp` (Vietnamese NLP)
- Pluggable embeddings: `gemini` (multi-key rotation) or `local` (sentence-transformers)
- Dual query engine:
  - API mode: Gemini Cypher generation + safety validation + hybrid fallback
  - Local mode: ReAct agent with kg_schema, kg_query, text_search, get_passage tools
- Dataset generation pipeline:
  - 2-hop and 3-hop KG walk extraction
  - Vietnamese question templates per entity type
  - LLM rewrite for naturalness (optional)
  - 3-stage QC: well-formedness, grounding, deduplication
- Reliability:
  - Persistent HF job state in `.hf_ingest_jobs.json`
  - Startup restore marks stale running jobs as `interrupted`
  - Atomic job-state writes
- Operations:
  - `/health`, `/ready`, `/metrics`
  - Optional API key auth + per-client rate limiting

## Quick start

### 1) Configure

```bash
cp .env.example .env
printf "%s" "YOUR_GEMINI_API_KEY" > .gemini_key.txt
chmod 600 .gemini_key.txt
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NER_BACKEND` | `simple` | `simple`, `underthesea`, or `phonlp` |
| `EMBEDDING_BACKEND` | `gemini` | `gemini` or `local` |
| `MODEL_MODE` | `api` | `api` (Gemini) or `local` (Qwen2.5-7B) |
| `LOCAL_MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model for local mode |

### 2) Start Neo4j

```bash
docker compose up -d
```

### 3) Install dependencies

```bash
uv sync --all-groups
```

### 4) Run API

```bash
uv run uvicorn src.main:app --reload --port 8000
```

Docs: <http://localhost:8000/docs>

## API examples

### Ingest topics

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"topics":["Graph database","Neo4j"]}'
```

### Ingest HF sample

```bash
curl -X POST "http://localhost:8000/ingest/hf" \
  -H "Content-Type: application/json" \
  -d '{"config_name":"20231101.vi","split":"train","sample_size":2,"streaming":true}'
```

### Start async HF job

```bash
curl -X POST "http://localhost:8000/ingest/hf/jobs" \
  -H "Content-Type: application/json" \
  -d '{"config_name":"20231101.vi","split":"train","sample_size":3,"streaming":true}'
```

### Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Neo4j được sử dụng như thế nào?","top_k":4}'
```

### Health/Readiness/Metrics

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/ready"
curl "http://localhost:8000/metrics"
```

## Dataset generation

For reproducible Vietnamese Wikipedia inputs, use the processed HF snapshot
[`Keithsel/viwiki-20260523`](https://huggingface.co/datasets/Keithsel/viwiki-20260523).
It is produced from raw MediaWiki XML by `scripts/viwiki_processing/`, which
exports both cleaned article text and raw wikitext Parquet shards.

Install XML processing dependencies with `uv sync --group xml-processing` before
running the raw dump converter.

After ingesting data, generate the ViWiki-MHR multi-hop QA dataset:

```python
from src.dataset_gen import generate_dataset

stats = generate_dataset(
    two_hop_limit=5000,
    three_hop_limit=3000,
    broken_limit=1000,
    output_path="data/viwiki_mhr.jsonl",
    rewrite=True,
    qc=True,
)
```

## Development

```bash
make install
make check
make docs
```

Or run commands directly:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
python -m compileall -q src tests
```

CI workflow is under `.github/workflows/ci.yml`.
