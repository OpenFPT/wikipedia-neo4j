# ViWiki-MHR Project Roadmap

## Discovery Summary

**Source of truth:** `knowledge-base/30-Synthesis/Project Gap Analysis and Rebuilding Strategy.md`

**Current state:** Core modules implemented (~60% of target). KG populated with 1000 Vietnamese pages, SLM runtime working, ReAct agent coded, dataset pipeline coded. Remaining: Vietnamese entity keywords, full dataset generation run, agent testing, QLoRA fine-tuning.

**Critical path:** Story 1 final task (Vietnamese entity keywords) → full dataset generation → QLoRA fine-tuning.

---

## Mode: Standard

Multiple interdependent workstreams, clear phases, but each phase is well-scoped with known techniques. No spike needed since the knowledge base already validates the approach.

---

## Epic Map (Capability Areas)

| Epic | Capability | Risk | Blocks |
|------|-----------|------|--------|
| **E1: Vietnamese KG Population** | Ingest vi-wiki, typed schema, validated NER | NER quality on Vietnamese text | E3, E4 |
| **E2: Local SLM Runtime** | Qwen2.5-7B in 4-bit NF4, dual-mode toggle | VRAM fit, inference speed | E3, E5 |
| **E3: ReAct Agent Loop** | 4 tools, 6-iter cap, citation tracking | SLM tool-calling reliability | E5 |
| **E4: ViWiki-MHR Dataset** | 36K examples, KG walks, adversarial unanswerables | QC pipeline quality | E5 |
| **E5: QLoRA Fine-tuning** | Text2Cypher adapter, agent format, DPO | Training stability, eval metrics | — |

```
E1 ──┬──► E3 ──► E5
     │         ▲
E2 ──┘         │
               │
E4 ────────────┘
```

---

## Current Story Pack (Active Work)

### Story 1: Vietnamese Wikipedia → Typed Neo4j Graph (E1)

**Exit state:** Neo4j chứa >=1000 pages Vietnamese Wikipedia với typed nodes (Person, Organization, Location, Work) và typed relationships.

- [x] Thêm `local_path` param vào `ingest_from_hf()` để load từ `data/viet-wikipedia`
- [x] Chuyển default config sang Vietnamese (local path)
- [x] Extend Neo4j schema: typed entity labels + typed mention edges (đã có sẵn trong code)
- [x] Validate underthesea/phonlp NER output trên Vietnamese text (spot-check 50 entities)
- [x] Chạy ingestion batch nhỏ (1000 pages), verify graph structure
- [ ] Thêm Vietnamese-specific entity classification keywords vào `_classify_entity_type()`

---

## Completed Stories

### Story 2: Local SLM Hello-World (E2) ✅

**Exit state:** Qwen2.5-7B-Instruct chạy 4-bit NF4 trên local machine, generate được text từ Vietnamese prompt.

- [x] Thêm `transformers`, `bitsandbytes`, `accelerate` vào dependencies
- [x] Tạo `src/local_llm.py` với load/generate wrapper
- [x] Implement `MODEL_MODE=local|api` toggle trong config
- [x] Test: input Vietnamese question → nhận output coherent

---

## In Progress Stories

### Story 3: ReAct Agent Loop (E3)

- [x] Implement 4 tools: `kg_schema()`, `kg_query()`, `text_search()`, `get_passage()`
- [x] Build Thought → Action → Observation loop, cap 6 iterations
- [x] Citation tracking + groundedness post-processor
- [ ] Test trên 100-question sample

### Story 4: ViWiki-MHR Dataset Generation (E4)

- [x] Offline pipeline: KG walks → template questions → LLM rewrite
- [ ] Target: ~8K synthetic multi-hop + ~28K reformatted UIT-ViQuAD 2.0
- [x] Adversarial unanswerables: broken-link walks (~1K)
- [x] 3-stage QC pipeline (grounding, NLI, well-formed filter)
- [ ] Human-verified test split (1000-2000 samples)

---

## Future Stories

### Story 5: QLoRA Fine-tuning (E5) — blocked by E3 + E4

- [ ] Text-to-Cypher adapter (~10-15K pairs)
- [ ] Agent action format conformance training
- [ ] DPO alignment pass
- [ ] Evaluation harness: joint EM/F1 on ViWiki-MHR test split

---

## Architecture Target

```
+-----------------------------------------------------------------------------------+
|                              Local Consumer Device                                |
|                                                                                   |
|  +--------------------+      +-------------------------------------------------+  |
|  |    User Prompt     | ---> |               SLM Orchestrator                  |  |
|  | (Vietnamese Query) |      | (Qwen 2.5-7B in 4-bit NF4 VRAM)                |  |
|  +--------------------+      +-------------------------------------------------+  |
|           ^                                           |                           |
|           |                                           v                           |
|  +--------------------+              +----------------------------------+         |
|  |    Final Answer    | <--- [Pass]  |    Citation & Groundedness       |         |
|  | + Verified Sources |              |        Post-Processor            |         |
|  +--------------------+              +----------------------------------+         |
|                                                       ^                           |
|                                                       | [Fail: Flag Ungrounded]   |
|                                                                                   |
|                   +---------------------------------------+                       |
|                   |           ReAct Agent Loop            |                       |
|                   | (Thought -> Action -> Observation)    |                       |
|                   |      Hard capped at 6 iterations      |                       |
|                   +---------------------------------------+                       |
|                                       |                                           |
|                   +-------------------+-------------------+                       |
|                   |                   |                   |                       |
|                   v                   v                   v                       |
|           +---------------+   +---------------+   +---------------+               |
|           |  kg_query()   |   | text_search() |   | get_passage() |               |
|           |  kg_schema()  |   |               |   |               |               |
|           +---------------+   +---------------+   +---------------+               |
|                   |                   |                   |                       |
|                   v                   v                   v                       |
|           +---------------+   +---------------+   +---------------+               |
|           |   Neo4j       |   | Qdrant/FAISS  |   | Wikipedia     |               |
|           |   (Graph KG)  |   |   + BM25      |   | Paragraph     |               |
|           +---------------+   +---------------+   | Local Dump    |               |
|                                                   +---------------+               |
+-----------------------------------------------------------------------------------+
```

---

## Key Design Decisions

- **Fully local, privacy-preserving** — no outbound network dependencies at runtime
- **Hardware target:** 16GB RAM, 8GB VRAM (RTX 3060/4060 class)
- **Dataset:** DataStudio/Viet-wikipedia (1.29M articles, local Arrow files at `data/viet-wikipedia/`)
- **NER:** underthesea/phonlp backends (already configured)
- **Base SLM:** Qwen2.5-7B-Instruct (multilingual, tool-calling capable)
- **Fine-tuning:** QLoRA (4-bit NF4, rank 32, alpha 64)
- **Evaluation:** ViWiki-MHR test split only (no ViMQA dependency)
