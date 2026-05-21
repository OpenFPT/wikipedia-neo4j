# Changelog

## v0.7.0 (current)

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
