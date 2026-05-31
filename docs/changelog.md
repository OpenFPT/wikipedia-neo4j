# Changelog

## v0.9.0 (current)

- Added WRRF hybrid retrieval combining BM25, vector, graph, and community signals (`src/retrieve.py`).
- Added community-based retrieval module with Louvain summaries (`src/community.py`).
- Added entity resolution for Vietnamese aliases and diacritic variants (`src/entity_resolution.py`).
- Added LLM-based typed relation extraction with 6 relation types (`src/relation_extract.py`).
- Added multi-trajectory agent execution with majority voting and question decomposition (`src/agent.py`).
- Added 2 new agent tools: `entity_neighborhood` and `path_search`.
- Added UIT-ViQuAD2.0 evaluation adapter achieving 72.6% hit rate (`src/viquad_adapter.py`).
- Added `phobert`, `videberta`, and `wikilink` NER backends.
- Added Gradio demo interface (`src/app_gradio.py`).
- Switched default model to AITeamVN/Vi-Qwen2-7B-RAG.
- Switched default embedding to GreenNode-Embedding-Large-VN-Mixed-V1 (1024-dim).
- Added CI parallel jobs, caching, and coverage gate.
- Fixed agent architecture (proper tool separation, prompt templates).
- Improved NER entity classification and export data quality.

## v0.8.0

- Added cross-encoder reranking module (`src/reranker.py`) using BAAI/bge-reranker-v2-m3.
- Added evaluation pipeline (`src/evaluation.py`) with context hit rate, MRR, and latency metrics.
- Completed Vietnamese Wikipedia ingestion (1000 pages) with typed entity graph.
- Validated underthesea/phonlp NER output on Vietnamese text.

## v0.7.0

- Added KG walk extraction and question template engine (`src/dataset_gen.py`).
- Added LLM rewrite stage for naturalizing template questions.
- Added 3-stage QC pipeline (well-formedness, grounding, deduplication).
- Added ViWiki-MHR dataset output in JSONL format.

## v0.6.0

- Added ReAct agent loop with 4 graph tools for multi-hop QA (`src/agent.py`).
- Added local SLM wrapper for Qwen2.5-7B-Instruct with 4-bit NF4 quantization (`src/local_llm.py`).
- Wired local model into Cypher generation pipeline.
- Added `MODEL_MODE` config: `api` (Gemini) or `local` (Qwen2.5).
- Extracted NER to dedicated module (`src/ner.py`) with BIO tag accumulation.
- Simplified entity Cypher and defaulted to Vietnamese wiki config.

## v0.5.0

- Added FastAPI lifespan lifecycle and runtime config validation.
- Added `/ready` and `/metrics` endpoints.
- Added optional API key auth and per-client rate limit guard.
- Added HF jobs list endpoint with pagination/filtering.
- Improved retrieval fallback to hybrid fulltext strategy.
- Expanded tests for HF ingestion and background jobs APIs.
- Added CI workflow with lint, type-check, tests, and compile checks.
- Added coverage threshold gate.
