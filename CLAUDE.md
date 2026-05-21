# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
uv sync --all-groups          # Install all dependencies
uv run uvicorn src.main:app --reload --port 8000  # Run dev server
docker compose up -d          # Start Neo4j (required for runtime)
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

This is a **GraphRAG system** that ingests Wikipedia content into a Neo4j knowledge graph and answers questions using graph-based retrieval.

### Data Flow

1. **Ingestion** (`src/ingest.py`): Wikipedia API or HF dataset → chunk text → extract entities (NER) → generate embeddings → write to Neo4j
2. **Query** (`src/retrieve.py`): Question → Gemini generates read-only Cypher → validate safety → execute → fallback to hybrid fulltext if generation fails

### Graph Schema

- `Page -[:HAS_CHUNK]-> Chunk -[:MENTIONS]-> Entity`
- `Page -[:LINKS_TO]-> Page`
- Typed entity labels: `Person`, `Organization`, `Location`, `Work`
- Typed mention edges: `MENTIONS_PERSON`, `MENTIONS_ORG`, `MENTIONS_LOCATION`, `MENTIONS_WORK`

### Key Design Decisions

- **NER is pluggable** via `NER_BACKEND` env: `simple` (regex), `underthesea`, or `phonlp` (Vietnamese NLP with VnCoreNLP word segmentation)
- **Embeddings are pluggable** via `EMBEDDING_BACKEND`: `gemini` (with multi-key rotation on rate-limit) or `local` (sentence-transformers)
- **Cypher generation safety**: LLM output is validated against a blocklist of write keywords and must return exact aliases (`page_title`, `page_url`, `chunk_id`, `chunk_text`, `score`)
- **Gemini keys** are stored in a plaintext file (default `.gemini_key.txt`), one key per line, for rotation
- **Background HF ingestion jobs** run in daemon threads with progress callbacks, cancellation via `threading.Event`, and persistent state in `.hf_ingest_jobs.json` (atomic writes via tmp+rename)
- **On restart**, stale running/cancelling jobs are marked `interrupted`

### Module Responsibilities

- `src/main.py` — FastAPI app, API key auth, rate limiting, job lifecycle, health/ready/metrics
- `src/llm.py` — Gemini client pool, embedding generation, Cypher generation + validation
- `src/neo4j_client.py` — Driver singleton, schema/index/constraint setup
- `src/job_store.py` — Thread-safe JSON file persistence for job state
- `src/config.py` — Pydantic Settings from `.env`, Gemini key loading

## Configuration

Copy `.env.example` to `.env`. Neo4j runs via docker-compose on `bolt://localhost:7687` with default auth `neo4j/please-change-me`.

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
