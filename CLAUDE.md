# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
uv sync --all-groups          # Install all dependencies
uv run uvicorn src.main:app --reload --port 8000  # Run dev server
sudo systemctl start neo4j    # Start Neo4j (required for runtime)
```

## Test & Lint

```bash
uv run pytest                 # Run all tests (75%+ coverage required)
uv run pytest tests/test_ingest_core.py -k "test_chunk"  # Single test
uv run ruff check src tests   # Lint
uv run mypy src               # Type check
make check                    # All three: lint + typecheck + test
```

Pre-commit hooks run ruff (with --fix) and ruff-format automatically.

## Architecture

This is a **GraphRAG system** that ingests Vietnamese Wikipedia content into a Neo4j knowledge graph, answers multi-hop questions using graph-based retrieval, and generates QA datasets.

### Data Flow

1. **Ingestion** (`src/ingest.py`): Wikipedia API or HF dataset → chunk text → extract entities via NER (`src/ner.py`) → generate embeddings → write to Neo4j
2. **Bulk Ingestion** (3-step pipeline):
   - `scripts/export_dataset.py`: Arrow dataset → filter stubs → Unicode normalize → chunk → NER → JSONL/CSV files
   - `scripts/embed_chunks.py`: chunks.jsonl → batch embeddings → chunk_embeddings.jsonl
   - `scripts/load_neo4j.py`: JSONL/CSV → batched UNWIND → Neo4j
3. **Query — API mode** (`src/retrieve.py`): Question → Gemini generates read-only Cypher → validate safety → execute → fallback to hybrid fulltext if generation fails
4. **Query — Local mode** (`src/agent.py`): Question → ReAct agent loop (up to 6 iterations) → graph tools (kg_schema, kg_query, text_search, get_passage) → final answer with citations
5. **Dataset generation** (`src/dataset_gen.py`): KG walk extraction → template QA → optional LLM rewrite → QC pipeline → JSONL output

### Graph Schema

- `Page -[:HAS_CHUNK]-> Chunk -[:MENTIONS]-> Entity`
- `Page -[:LINKS_TO]-> Page`
- Typed entity labels: `Person`, `Organization`, `Location`, `Work`
- Typed mention edges: `MENTIONS_PERSON`, `MENTIONS_ORG`, `MENTIONS_LOCATION`, `MENTIONS_WORK`

### Key Design Decisions

- **NER is pluggable** via `NER_BACKEND` env: `simple` (regex + keyword classification), `underthesea` (BIO tagging), or `phonlp` (PhoNLP + VnCoreNLP word segmentation)
- **Embeddings are pluggable** via `EMBEDDING_BACKEND`: `gemini` (with multi-key rotation on rate-limit) or `local` (sentence-transformers)
- **Model mode** via `MODEL_MODE`: `api` (Gemini for Cypher generation) or `local` (Qwen2.5-7B-Instruct, 4-bit NF4 quantized, for ReAct agent)
- **Cypher generation safety**: LLM output is validated against a blocklist of write keywords and must return exact aliases (`page_title`, `page_url`, `chunk_id`, `chunk_text`, `score`)
- **Gemini keys** are stored in a plaintext file (default `.gemini_key.txt`), one key per line, for rotation
- **Background HF ingestion jobs** run in daemon threads with progress callbacks, cancellation via `threading.Event`, and persistent state in `.hf_ingest_jobs.json` (atomic writes via tmp+rename)
- **On restart**, stale running/cancelling jobs are marked `interrupted`
- **Dataset generation** uses KG walks (2-hop, 3-hop, broken-link) with Vietnamese question templates, optional LLM rewrite for naturalness, and 3-stage QC (well-formedness, grounding, dedup)

### Module Responsibilities

- `src/main.py` — FastAPI app, API key auth, rate limiting, job lifecycle, health/ready/metrics
- `src/ingest.py` — Wikipedia API and HF ingestion pipelines
- `src/ner.py` — Pluggable NER backends, BIO tag accumulation, entity type classification
- `src/retrieve.py` — Cypher-based retrieval with hybrid fulltext fallback
- `src/agent.py` — ReAct agent loop with 4 graph tools for multi-hop QA
- `src/llm.py` — Gemini client pool, embedding generation (single + batch), Cypher generation + validation
- `src/local_llm.py` — Local SLM wrapper (Qwen2.5-7B-Instruct, 4-bit NF4, lazy-loaded)
- `src/dataset_gen.py` — KG walk extraction, question templates, LLM rewrite, QC pipeline
- `src/neo4j_client.py` — Driver singleton, schema/index/constraint setup, batch UNWIND writes
- `src/job_store.py` — Thread-safe JSON file persistence for job state
- `src/config.py` — Pydantic Settings from `.env`, Gemini key loading, runtime validation
- `src/logging_utils.py` — Structured logging with request-ID context and file-based log output
- `src/text_utils.py` — Vietnamese Unicode normalization and text chunking utilities
- `src/reranker.py` — Cross-encoder reranking (BAAI/bge-reranker-v2-m3)
- `src/evaluation.py` — Evaluation pipeline: context hit rate, MRR, latency on ViWiki-MHR

## Configuration

Copy `.env.example` to `.env`. Neo4j runs as a systemd service on `bolt://localhost:7687`.

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NER_BACKEND` | `simple` | NER engine: `simple`, `underthesea`, `phonlp` |
| `EMBEDDING_BACKEND` | `gemini` | Embedding engine: `gemini`, `local` |
| `MODEL_MODE` | `api` | Query engine: `api` (Gemini), `local` (Qwen2.5) |
| `LOCAL_MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model ID for local mode |
| `APP_API_KEY` | (none) | Optional API key for protected endpoints |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per-client rate limit |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `JSON_LOGS` | `false` | Structured JSON log output |
| `LOG_DIR` | `logs` | Directory for file-based log output |
| `MIN_TEXT_LENGTH` | `200` | Minimum article length for bulk ingestion |
| `INGEST_BATCH_SIZE` | `100` | Articles per processing batch |
| `EMBED_BATCH_SIZE` | `50` | Chunks per embedding API call |

## Bulk Ingestion Pipeline

For large-scale ingestion of the ViWiki dataset (1.6M articles, ~590K useful):

```bash
# 1. Download dataset (one-time)
uv run python scripts/download_dataset.py

# 2. Export to JSONL/CSV (filters stubs, runs NER, ~2-4h for full dataset)
uv run python scripts/export_dataset.py

# 3. Generate embeddings (optional, resumable)
uv run python scripts/embed_chunks.py --backend local

# 4. Load into Neo4j (batched UNWIND, ~30-60 min)
uv run python scripts/load_neo4j.py --drop-indexes

# 5. Verify integrity
uv run python scripts/verify_ingestion.py
```

Intermediate files are stored in `data/export/` (JSONL/CSV, inspectable with jq/head).
All scripts support `--limit N` for testing, checkpoint-based resumability, and SIGINT.
Logs are written to `logs/{task}_{timestamp}.log`.

## ClaudeVibeCodeKit

### Planning
When planning complex tasks:
1. Read `.claude/docs/plan-execution-guide.md` for format guide
2. Use planning-agent for parallel execution optimization
3. Output plan according to `.claude/schemas/plan-schema.json`

### Available Commands
- `/research <topic>` - Deep web research
- `/meeting-notes <name>` - Live meeting notes
- `/changelog` - Generate changelog
- `/onboard` - Developer onboarding
- `/handoff` - Create handoff document for conversation transition
- `/continue` - Resume work from a handoff document
- `/watzup` - Check current project status
- `/social-media-post` - Social content workflow
