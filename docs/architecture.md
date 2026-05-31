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
- `NER_BACKEND=phobert` uses PhoBERT transformer NER pipeline.
- `NER_BACKEND=videberta` uses ViDeBERTa (NlpHUST/ner-vietnamese-electra-base).
- `NER_BACKEND=wikilink` uses Wikipedia hyperlinks for entity extraction (best for bulk ingestion, Typed F1=46.9%).

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
- `src/ner.py`: Pluggable NER backends (simple/underthesea/phonlp/phobert/videberta/wikilink) and entity type classification
- `src/retrieve.py`: WRRF hybrid retrieval (BM25 + vector + graph + community fusion), reranking, legacy Cypher generation fallback
- `src/agent.py`: ReAct agent with 6 tools, complexity detection, question decomposition, multi-trajectory voting
- `src/llm.py`: Gemini client pool, embedding generation, Cypher generation/validation
- `src/local_llm.py`: Local SLM wrapper (Qwen2.5-7B-Instruct, 4-bit NF4 quantization)
- `src/dataset_gen.py`: KG walk extraction, question template engine, LLM rewrite, and QC pipeline
- `src/neo4j_client.py`: Neo4j driver + schema/index setup + connectivity check
- `src/job_store.py`: Persistent JSON store for async HF jobs
- `src/config.py`: Pydantic Settings from `.env`, Gemini key loading, runtime validation
- `src/logging_utils.py`: Structured logging with request-ID context
- `src/reranker.py`: Cross-encoder reranking (BAAI/bge-reranker-v2-m3), used in WRRF fusion pipeline
- `src/evaluation.py`: Evaluation pipeline — context hit rate, MRR, latency on ViWiki-MHR dataset; ViQuAD2.0 support (72.6% hit rate)
- `src/agent_tools.py`: Agent tool definitions (kg_schema, kg_query, text_search, get_passage, entity_neighborhood, path_search)
- `src/prompts.py`: Prompt templates for agent, Cypher generation, and rewriting
- `src/community.py`: Community-based retrieval — Louvain membership lookup, pre-generated summaries, chunk retrieval
- `src/entity_resolution.py`: Vietnamese entity resolution — diacritic normalization, alias merging, canonical form lookup
- `src/relation_extract.py`: LLM-based typed relation extraction (6 types: FOUNDED_BY, LOCATED_IN, BORN_IN, MEMBER_OF, PART_OF, CREATED_BY)
- `src/viquad_adapter.py`: UIT-ViQuAD2.0 HuggingFace adapter for evaluation benchmarking
- `src/app_gradio.py`: Gradio demo interface for interactive QA
- `scripts/viwiki_processing/`: raw MediaWiki XML streaming, wikitext cleanup, and cleaned/raw Parquet export for `Keithsel/viwiki-20260523`

## Query behavior

Two query modes controlled by `MODEL_MODE`:

### API mode (`MODEL_MODE=api`, default)

1. Primary path: WRRF hybrid retrieval (BM25 + vector + graph + community fusion) with cross-encoder reranking.
2. Legacy fallback: Gemini-generated read-only Cypher, validated for safety (write-keyword blocklist) and output aliases.
3. Execute against Neo4j.
4. On invalid generation/runtime shape failure, fallback to hybrid fulltext retrieval.

### Local agent mode (`MODEL_MODE=local`)

1. Complexity detection classifies the incoming question; complex queries trigger question decomposition into sub-questions.
2. ReAct agent loop (`src/agent.py`) with up to 6 iterations.
3. Agent has 6 tools: `kg_schema`, `kg_query`, `text_search`, `get_passage`, `entity_neighborhood`, `path_search`.
4. Agent reasons step-by-step, calling tools and collecting observations.
5. For complex queries, multi-trajectory execution runs parallel reasoning paths with majority voting for the final answer.
6. On convergence, returns final answer with citations.
7. On timeout, synthesizes answer from collected observations.

## Hybrid retrieval (WRRF)

WRRF (Weighted Reciprocal Rank Fusion) is the primary retrieval strategy in API mode. It combines 4 signals:

1. **BM25 fulltext**: Neo4j fulltext index search over chunk text.
2. **Vector similarity**: Cosine similarity on chunk embeddings.
3. **Graph traversal**: Multi-hop entity-linked chunk discovery via graph relationships.
4. **Community-based retrieval**: Louvain community membership lookup with pre-generated summaries.

Configurable weights via environment variables: `WRRF_WEIGHT_BM25`, `WRRF_WEIGHT_VECTOR`, `WRRF_WEIGHT_GRAPH`, `WRRF_WEIGHT_COMMUNITY`.

After fusion, cross-encoder reranking (`src/reranker.py`, BAAI/bge-reranker-v2-m3) is applied to the merged candidate set before returning final results.

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
