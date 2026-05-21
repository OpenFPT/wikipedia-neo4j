# Wikipedia Neo4j GraphRAG Demo

Backend service for ingesting Vietnamese Wikipedia content into a Neo4j knowledge graph, answering multi-hop questions using graph-based retrieval, and generating QA datasets.

## Key capabilities

- Topic ingestion via Wikipedia API
- Dataset ingestion via Hugging Face (`wikimedia/wikipedia`)
- Async background HF ingestion jobs with persistent state
- Pluggable NER: simple regex, Underthesea, or PhoNLP (Vietnamese)
- Pluggable embeddings: Gemini API (with multi-key rotation) or local sentence-transformers
- Gemini-assisted read-only Cypher generation with safety validation
- ReAct agent loop with graph tools for multi-hop QA (local model mode)
- Local SLM support: Qwen2.5-7B-Instruct (4-bit quantized)
- Multi-hop QA dataset generation from KG walks (ViWiki-MHR)
- Hybrid fallback retrieval for robust answers
- Health, readiness, and metrics endpoints

## Quick links

- [Architecture](architecture.md)
- [Setup & Run](setup.md)
- [API Endpoints](api/endpoints.md)
- [Background Jobs](api/background-jobs.md)
- [Ingestion: Wikipedia API](ingestion-wikipedia.md)
- [Ingestion: Hugging Face](ingestion-hf.md)
- [Operations & Troubleshooting](operations.md)


## Operating modes

- **Ingest**: populate graph data from Wikipedia API and HF datasets.
- **Query (API mode)**: Gemini generates Cypher, with hybrid fulltext fallback.
- **Query (Local mode)**: ReAct agent iterates over graph tools using a local LLM.
- **Dataset generation**: extract KG walks and produce multi-hop QA pairs.
- **Jobs**: run long HF ingestions asynchronously.
- **Ops**: health, readiness, and metrics endpoints for reliability.
