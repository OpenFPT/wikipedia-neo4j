# Project Progress

**Last updated:** 2026-05-27  
**Current branch:** `main`  
**Roadmap:** See [plans/roadmap.md](../plans/roadmap.md)

---

## Current Status

### Story 1: Vietnamese Wikipedia → Typed Neo4j Graph (E1) — 95%

**Progress:** 6/6 core tasks completed, entity backfill in progress

- [x] Thêm `local_path` param vào `ingest_from_hf()`
- [x] Chuyển default config sang Vietnamese (local path)
- [x] Extend Neo4j schema: typed entity labels + typed mention edges
- [x] Validate underthesea/phonlp NER output trên Vietnamese text
- [x] Chạy ingestion batch nhỏ (1000 pages), verify graph structure
- [x] Thêm Vietnamese-specific entity classification keywords vào `_classify_entity_type()`
- [x] Added phobert, videberta, wikilink NER backends
- [ ] Entity backfill and verification (in progress)

### Story 2: Local SLM Hello-World (E2) — 100% ✅

- [x] Thêm `transformers`, `bitsandbytes`, `accelerate` vào dependencies
- [x] Tạo `src/local_llm.py` với load/generate wrapper
- [x] Implement `MODEL_MODE=local|api` toggle trong config
- [x] Test: input Vietnamese question → nhận output coherent
- [x] Switched to AITeamVN/Vi-Qwen2-7B-RAG

### Story 3: ReAct Agent Loop (E3) — 95%

- [x] Implement 6 tools: kg_schema, kg_query, text_search, get_passage, entity_neighborhood, path_search
- [x] Build Thought → Action → Observation loop, cap 6 iterations
- [x] Citation tracking + groundedness post-processor
- [x] Complexity detection and question decomposition
- [x] Multi-trajectory execution with majority voting
- [x] Agent architecture audit and fixes
- [ ] Full evaluation on 100-question sample

### Story 4: ViWiki-MHR Dataset Generation (E4) — 70%

- [x] Offline pipeline: KG walks → template questions → LLM rewrite
- [ ] Target: ~8K synthetic multi-hop + ~28K reformatted UIT-ViQuAD 2.0
- [x] Adversarial unanswerables: broken-link walks (~1K)
- [x] 3-stage QC pipeline (grounding, NLI, well-formed filter)
- [ ] Human-verified test split (1000-2000 samples)

### Story 5: Hybrid Retrieval & Graph Enrichment (E5) — 80%

- [x] WRRF hybrid retrieval (BM25 + vector + graph + community)
- [x] Cross-encoder reranking (BAAI/bge-reranker-v2-m3)
- [x] Community detection with Louvain summaries
- [x] Entity resolution (Vietnamese aliases, diacritics)
- [x] LLM-based relation extraction (6 typed relations)
- [ ] Full-scale community summary generation
- [ ] Relation extraction on full corpus

### Story 6: Evaluation & Benchmarking (E6) — 60%

- [x] Evaluation pipeline: hit rate, MRR, latency
- [x] UIT-ViQuAD2.0 adapter (72.6% hit rate)
- [x] CI coverage gate (75%+)
- [ ] End-to-end EM/F1 on ViWiki-MHR test split
- [ ] Ablation studies (retrieval components)

### Story 7: QLoRA Fine-tuning (E7) — 0%

- [ ] Text-to-Cypher adapter (~10-15K pairs)
- [ ] Agent action format conformance training
- [ ] DPO alignment pass
- [ ] Evaluation harness: joint EM/F1 on ViWiki-MHR test split

---

## Overall Progress

| Epic | Status | Progress |
|------|--------|----------|
| E1: Vietnamese KG Population | ✅ Near complete | 95% |
| E2: Local SLM Runtime | ✅ Done | 100% |
| E3: ReAct Agent Loop | ✅ Near complete | 95% |
| E4: ViWiki-MHR Dataset | 🚧 Pipeline done, needs data | 70% |
| E5: Hybrid Retrieval & Enrichment | 🚧 Core done | 80% |
| E6: Evaluation & Benchmarking | 🚧 In progress | 60% |
| E7: QLoRA Fine-tuning | ⏳ Not started | 0% |
