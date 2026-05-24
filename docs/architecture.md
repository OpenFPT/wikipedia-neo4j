# Architecture

## Data model

- `(:Page {id, title, url, summary})`
- `(:Chunk {id, text, sequence_number, embedding})`
- `(:Entity {id, name, type})`
- `(:Person {id, name, type})` (typed entity label)
- `(:Organization {id, name, type})` (typed entity label)
- `(:Location {id, name, type})` (typed entity label)
- `(:Work {id, name, type})` (typed entity label)

Entity extraction (`src/ner.py`):

- `NER_BACKEND=simple` uses a regex heuristic (title-cased words) + keyword-based type classification.
- `NER_BACKEND=underthesea` uses Underthesea NER (offline) with BIO tag accumulation.
- `NER_BACKEND=phonlp` uses PhoNLP + VnCoreNLP word segmentation (offline) with BIO tag accumulation.

All backends return `(name, type)` tuples. Type classification maps NER tags (`PER`, `ORG`, `LOC`, `MISC`) to graph labels, with keyword fallback for the simple backend.

Relationships:

- `(:Page)-[:HAS_CHUNK]->(:Chunk)`
- `(:Chunk)-[:MENTIONS]->(:Entity)`
- `(:Chunk)-[:MENTIONS_PERSON]->(:Person)`
- `(:Chunk)-[:MENTIONS_ORG]->(:Organization)`
- `(:Chunk)-[:MENTIONS_LOCATION]->(:Location)`
- `(:Chunk)-[:MENTIONS_WORK]->(:Work)`
- `(:Page)-[:LINKS_TO]->(:Page)`

## Components

- `src/main.py`: FastAPI app, auth/rate-limit guard, job APIs, health/readiness/metrics
- `src/ingest.py`: Wikipedia API, Hugging Face, and local dataset ingestion pipelines
- `src/ner.py`: Pluggable NER backends (simple/underthesea/phonlp) and entity type classification
- `src/retrieve.py`: Retrieval and answer assembly
- `src/agent.py`: ReAct agent loop with graph tools for multi-hop QA
- `src/llm.py`: Gemini client pool, embedding generation, Cypher generation/validation
- `src/local_llm.py`: Local SLM wrapper (Qwen2.5-7B-Instruct, 4-bit NF4 quantization)
- `src/dataset_gen.py`: KG walk extraction, question template engine, LLM rewrite, and QC pipeline
- `src/neo4j_client.py`: Neo4j driver + schema/index setup + connectivity check
- `src/job_store.py`: Persistent JSON store for async HF jobs
- `src/config.py`: Pydantic Settings from `.env`, Gemini key loading, runtime validation
- `src/logging_utils.py`: Structured logging with request-ID context
- `src/reranker.py`: Cross-encoder reranking (BAAI/bge-reranker-v2-m3) for retrieval results
- `src/evaluation.py`: Evaluation pipeline — context hit rate, MRR, latency on ViWiki-MHR dataset
- `scripts/viwiki_processing/`: raw MediaWiki XML streaming, wikitext cleanup, and cleaned/raw Parquet export for `Keithsel/viwiki-20260523`

## Query behavior

Two query modes controlled by `MODEL_MODE`:

### API mode (`MODEL_MODE=api`, default)

1. Attempt Gemini-generated read-only Cypher.
2. Validate safety (write-keyword blocklist) and output aliases.
3. Execute against Neo4j.
4. On invalid generation/runtime shape failure, fallback to hybrid fulltext retrieval.

### Local agent mode (`MODEL_MODE=local`)

1. ReAct agent loop (`src/agent.py`) with up to 6 iterations.
2. Agent has 4 tools: `kg_schema`, `kg_query`, `text_search`, `get_passage`.
3. Agent reasons step-by-step, calling tools and collecting observations.
4. On convergence, returns final answer with citations.
5. On timeout, synthesizes answer from collected observations.

## Local model

When `MODEL_MODE=local`, the system uses Qwen2.5-7B-Instruct loaded with 4-bit NF4 quantization via bitsandbytes. The model is lazy-loaded on first call and cached in memory.

Provides two interfaces:
- `generate(prompt)` — raw text completion
- `chat(messages)` — chat-template-aware generation (used by agent and dataset rewrite)

## Dataset generation pipeline

Raw corpus preparation starts from the Vietnamese Wikipedia MediaWiki XML dump.
`scripts/viwiki_processing/` streams the XML, filters main-namespace articles,
cleans wikitext into plain text, and exports both cleaned and raw Parquet shards.
The cleaned shard is suitable for graph ingestion and text retrieval, while the
raw shard preserves wikitext for later link/template extraction. The processed
snapshot is published as `Keithsel/viwiki-20260523` on Hugging Face for
reproducible downstream dataset generation and evaluation.

`src/dataset_gen.py` generates the ViWiki-MHR multi-hop QA dataset from the knowledge graph:

1. **KG walk extraction**: 2-hop and 3-hop paths through the graph, plus broken-link walks for unanswerable questions.
2. **Template-based QA**: Vietnamese question templates per entity type and hop count.
3. **LLM rewrite** (optional): Rewrites template questions using the local model for naturalness.
4. **QC pipeline**: 3-stage quality control (well-formedness, grounding check, deduplication).
5. **Output**: JSONL file with question, answer, and provenance metadata.

## Lifecycle

The app uses FastAPI lifespan for startup/shutdown tasks.
Shutdown closes shared Neo4j driver cleanly.
