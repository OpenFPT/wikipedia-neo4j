# 3. Requirements Specification

## 3.1 Functional Requirements

### Must-Have (M) — Core functionality required for project success

| ID | Requirement | Description | Verifiable By |
|----|------------|-------------|---------------|
| FR-01 | Wikipedia Ingestion | Ingest Vietnamese Wikipedia articles into Neo4j knowledge graph with Page, Chunk, Entity nodes and relationships | Test: `MATCH (p:Page) RETURN count(p)` > 0; integration test verifies node creation |
| FR-02 | Named Entity Recognition | Extract typed entities (Person, Organization, Location, Work) from article text | Test: NER F1 > 40% on typed entities; unit test with known articles |
| FR-03 | Embedding Generation | Generate 1024-dim semantic vectors for all chunks | Test: vector index exists in Neo4j; `CALL db.index.vector.queryNodes()` returns results |
| FR-04 | Hybrid Retrieval (WRRF) | Combine BM25 + vector + graph traversal retrieval signals using Weighted Reciprocal Rank Fusion | Test: MRR > 0.45 on ViWiki-MHR (beats any single signal) |
| FR-05 | Multi-hop QA | Answer questions requiring reasoning across 2+ articles, with supporting fact citations | Test: eval on ViWiki-MHR dataset; answers include `supporting_facts` field |
| FR-06 | REST API | `POST /query` endpoint accepting JSON `{"question": "..."}` and returning structured answer | Test: HTTP 200 with valid JSON schema; automated API test |
| FR-07 | Evaluation Pipeline | Compute Context Hit Rate, MRR, latency on benchmark datasets | Test: `uv run python -m src.evaluation --limit 10` completes without error |

### Should-Have (S) — Significant value, not blocking MVP

| ID | Requirement | Description | Verifiable By |
|----|------------|-------------|---------------|
| FR-08 | Community Retrieval | Louvain community detection provides topic-level retrieval signal | Test: community weight contributes to WRRF score; ablation shows improvement |
| FR-09 | Cross-encoder Reranking | BAAI/bge-reranker-v2-m3 reranks top-K results for precision | Test: rerank_MRR > base_MRR on eval set |
| FR-10 | Multi-trajectory Agent | N parallel reasoning trajectories with majority voting for complex questions | Test: accuracy improves with N=3 vs N=1 on hard questions |
| FR-11 | Question Decomposition | Automatically decompose complex multi-hop questions into sub-questions | Test: decomposed questions are answerable individually |
| FR-12 | Background Ingestion Jobs | Async HF ingestion with progress tracking, cancellation, persistence | Test: job state survives server restart; cancellation works |

### Could-Have (C) — Nice to have if time permits

| ID | Requirement | Description | Verifiable By |
|----|------------|-------------|---------------|
| FR-13 | Gradio Demo UI | Interactive web interface for QA demonstration | Test: `uv run python src/app_gradio.py` launches; can submit questions |
| FR-14 | MHQA Dataset Generation | Generate 700-1000 multi-hop QA pairs from KG walks + LLM | Test: output matches `template.json` schema; passes 5-layer QC |
| FR-15 | LoRA Fine-tuning Support | Load LoRA adapter for fine-tuned local model | Test: model loads with adapter; inference produces valid output |
| FR-16 | Entity Resolution | Merge diacritic variants and known aliases (e.g., "Bác Hồ" → "Hồ Chí Minh") | Test: query for alias returns canonical entity |

### Won't-Have (W) — Explicitly excluded

| ID | Requirement | Reason |
|----|------------|--------|
| FR-W1 | Real-time sync with Wikipedia | Out of scope — batch ingestion sufficient for research |
| FR-W2 | Multi-language support | Vietnamese-only by design decision |
| FR-W3 | User authentication/accounts | Research prototype, API key auth sufficient |
| FR-W4 | Horizontal scaling | Single-server deployment acceptable |

---

## 3.2 Non-Functional Requirements

### Performance

| ID | Requirement | Target | Measurement |
|----|------------|--------|-------------|
| NFR-01 | Simple query latency | < 5 seconds (P95) | Timed eval on 50+ queries |
| NFR-02 | Complex multi-hop latency | < 15 seconds (P95) | Timed eval on multi-hop set |
| NFR-03 | Context Hit Rate | > 70% on ViWiki-MHR | `src/evaluation.py` pipeline |
| NFR-04 | MRR (Mean Reciprocal Rank) | > 0.50 on ViWiki-MHR | `src/evaluation.py` pipeline |
| NFR-05 | Ingestion throughput | > 100 articles/minute (bulk) | Timed `scripts/load_neo4j.py` |
| NFR-06 | Embedding throughput | > 50 chunks/batch | Timed `scripts/embed_chunks.py` |

### Reliability

| ID | Requirement | Target | Measurement |
|----|------------|--------|-------------|
| NFR-07 | API uptime | > 99% during demo/eval sessions | Health endpoint monitoring |
| NFR-08 | Graceful degradation | System falls back to BM25 if vector/graph unavailable | Test: disable vector index, verify BM25 fallback |
| NFR-09 | Job persistence | Ingestion job state survives server restart | Test: kill server mid-job, restart, verify state |

### Security

| ID | Requirement | Target | Measurement |
|----|------------|--------|-------------|
| NFR-10 | API authentication | API key required for protected endpoints | Test: request without key returns 401 |
| NFR-11 | Cypher injection prevention | Block write operations in generated Cypher | Test: inject `DELETE` in question, verify blocked |
| NFR-12 | Rate limiting | 120 requests/minute per client | Test: exceed limit, verify 429 response |

### Maintainability

| ID | Requirement | Target | Measurement |
|----|------------|--------|-------------|
| NFR-13 | Test coverage | > 75% line coverage | `uv run pytest --cov` |
| NFR-14 | Code quality | Zero ruff lint errors | `uv run ruff check src tests` |
| NFR-15 | Type safety | No mypy errors on `src/` | `uv run mypy src` |
