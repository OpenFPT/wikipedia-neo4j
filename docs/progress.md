# Project Progress

**Last updated:** 2026-05-22  
**Current branch:** `fix/critical-bugs-and-coverage`  
**Roadmap:** See [plans/roadmap.md](../plans/roadmap.md)

---

## Current Status

### Story 1: Vietnamese Wikipedia → Typed Neo4j Graph (E1) — 83%

**Progress:** 5/6 tasks completed

- [x] Thêm `local_path` param vào `ingest_from_hf()`
- [x] Chuyển default config sang Vietnamese (local path)
- [x] Extend Neo4j schema: typed entity labels + typed mention edges
- [x] Validate underthesea/phonlp NER output trên Vietnamese text
- [x] Chạy ingestion batch nhỏ (1000 pages), verify graph structure
- [ ] Thêm Vietnamese-specific entity classification keywords vào `_classify_entity_type()`

### Story 2: Local SLM Hello-World (E2) — 100% ✅

- [x] Thêm `transformers`, `bitsandbytes`, `accelerate` vào dependencies
- [x] Tạo `src/local_llm.py` với load/generate wrapper
- [x] Implement `MODEL_MODE=local|api` toggle trong config
- [x] Test: input Vietnamese question → nhận output coherent

### Story 3: ReAct Agent Loop (E3) — 75%

- [x] Implement 4 tools: `kg_schema()`, `kg_query()`, `text_search()`, `get_passage()`
- [x] Build Thought → Action → Observation loop, cap 6 iterations
- [x] Citation tracking + groundedness post-processor
- [ ] Test trên 100-question sample

### Story 4: ViWiki-MHR Dataset Generation (E4) — 60%

- [x] Offline pipeline: KG walks → template questions → LLM rewrite
- [ ] Target: ~8K synthetic multi-hop + ~28K reformatted UIT-ViQuAD 2.0
- [x] Adversarial unanswerables: broken-link walks (~1K)
- [x] 3-stage QC pipeline (grounding, NLI, well-formed filter)
- [ ] Human-verified test split (1000-2000 samples)

### Story 5: QLoRA Fine-tuning (E5) — 0%

- [ ] Text-to-Cypher adapter (~10-15K pairs)
- [ ] Agent action format conformance training
- [ ] DPO alignment pass
- [ ] Evaluation harness: joint EM/F1 on ViWiki-MHR test split

---

## Overall Progress

| Epic | Status | Progress |
|------|--------|----------|
| E1: Vietnamese KG Population | 🚧 Near complete | 83% |
| E2: Local SLM Runtime | ✅ Done | 100% |
| E3: ReAct Agent Loop | 🚧 Code done, needs testing | 75% |
| E4: ViWiki-MHR Dataset | 🚧 Pipeline done, needs data | 60% |
| E5: QLoRA Fine-tuning | ❌ Not started | 0% |

**Estimated overall:** ~60%

---

## Architecture Status

### ✅ Implemented

- FastAPI backend với auth/rate-limit
- Neo4j driver + schema setup (systemd service)
- Pluggable NER backends (simple/underthesea/phonlp)
- Pluggable embedding backends (gemini/local)
- Wikipedia API ingestion
- HF dataset ingestion với async jobs
- Typed entity schema (Person, Organization, Location, Work)
- Typed mention relationships
- ReAct agent loop với 4 graph tools
- Local SLM (Qwen2.5-7B-Instruct, 4-bit NF4)
- Dataset generation pipeline (KG walks, templates, LLM rewrite, QC)
- Cross-encoder reranking (bge-reranker-v2-m3)
- Evaluation pipeline (context hit rate, MRR, latency)
- Health/readiness/metrics endpoints
- Structured logging với request-ID context

### 🚧 Needs Work

- Vietnamese-specific entity classification keywords
- Full dataset generation run (~36K examples)
- Human-verified test split
- Agent testing on 100-question sample

### ❌ Not Yet Implemented

- QLoRA fine-tuning pipeline
- Text-to-Cypher adapter training
- DPO alignment

---

## Blocking Dependencies

```
E1 (1 task left) ──┬──► E3 testing ──► E5
                   │                    ▲
E2 (done) ─────────┘                    │
                                        │
E4 (needs data gen) ───────────────────┘
```

---

## Next Steps

1. Complete Story 1: Vietnamese-specific entity classification keywords
2. Run full dataset generation (Story 4)
3. Test agent on 100-question sample (Story 3)
4. Begin Story 5: QLoRA fine-tuning

---

## References

- **Roadmap:** [plans/roadmap.md](../plans/roadmap.md)
- **Architecture:** [architecture.md](architecture.md)
- **Setup:** [setup.md](setup.md)
- **Operations:** [operations.md](operations.md)
