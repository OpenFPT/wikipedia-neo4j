# Next Phase Design: Eval-First Improvement Cycle

**Date:** 2026-05-31
**Status:** Draft
**Goal:** Close the Token F1 gap (0.12 → 0.40+), prove hybrid retrieval superiority via ablation, refactor large files, and build demo confidence — prioritized by defense value per effort.

---

## Phase 1: Full-Stack Ablation Study

### Dimensions

| Dimension | Variants |
|-----------|----------|
| NER backend | `simple`, `underthesea`, `phonlp`, `phobert`, `videberta`, `wikilink` |
| Retrieval | `bm25_only`, `vector_only`, `graph_only`, `wrrf_hybrid` |
| Generation | `base_prompt`, `improved_prompt`, `fine_tuned` (Phase 3) |

### Eval Dataset
- ViQuAD2 validation: 2652 questions with gold answers
- Subset for NER comparison: 100 representative pages re-ingested per backend

### Metrics
- Context Hit Rate (top-5)
- MRR
- Token F1
- Exact Match
- Average Latency (ms)

### Output
- `reports/ablation_results.json` — raw results per combination
- `reports/ablation_table.tex` — LaTeX-ready comparison table for slides
- `scripts/run_ablation.py` — single entry point, configurable dimensions

### NER Backend Comparison
Requires re-ingesting ~100 pages with each backend into isolated Neo4j databases (or namespaced labels). Compare downstream retrieval quality per backend.

---

## Phase 2: Refactoring Large Files

### agent.py (911 lines) → 4 modules

| New file | Responsibility |
|----------|---------------|
| `src/orchestration/agent_loop.py` | Core ReAct loop (think → act → observe) |
| `src/orchestration/complexity.py` | Question complexity detection |
| `src/orchestration/decomposition.py` | Multi-hop question decomposition |
| `src/orchestration/voting.py` | Multi-trajectory execution + majority voting |

Public interface unchanged: `run_agent()` delegates internally.

### hybrid.py (702 lines) → 5 modules

| New file | Responsibility |
|----------|---------------|
| `src/retrieval/bm25.py` | BM25 fulltext search |
| `src/retrieval/vector.py` | Dense vector similarity |
| `src/retrieval/graph.py` | Cypher graph traversal |
| `src/retrieval/community.py` | Community-based retrieval |
| `src/retrieval/fusion.py` | WRRF fusion + reranking orchestration |

Public interface unchanged: `hybrid_retrieve()` delegates internally.

### Principles
- No behavior change — pure structural refactor
- Each signal becomes independently testable and toggleable (enables ablation)
- Tests stay green throughout

---

## Phase 3: Closing the Token F1 Gap

### Step 1: Free Wins (Local, No Training)

1. **Prompt engineering:** Force short-span answers ("Answer in as few words as possible, using only words from the context")
2. **Post-processing:** Extract minimal answer span from verbose SLM output
3. **Expected improvement:** F1 0.12 → 0.25–0.35

### Step 2: QLoRA Fine-tune (Google Colab Free, T4 16GB)

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-7B-Instruct |
| Quantization | 4-bit NF4, double quant |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Target modules | All linear |
| Training data | ViQuAD2 train (~23K) + ViWiki-MHR multi-hop (~7K) |
| Task | Context + Question → short answer span |
| Eval | 500 held-out pairs |
| Target | Token F1 > 0.40 |

### Step 3: Re-run Ablation
- Add `fine_tuned` generation variant
- Show improvement: base_prompt → improved_prompt → fine_tuned
- Update thesis slides with final numbers

---

## Phase 4: Demo Confidence

### Golden Query Set (20-30 queries)

| Category | Count | Purpose |
|----------|-------|---------|
| Single-hop factual | 8 | Bread and butter |
| Multi-hop bridge | 6 | Graph reasoning showcase |
| Unanswerable | 4 | Abstain capability |
| Entity neighborhood | 4 | Graph traversal |
| Comparison / fan-out | 4 | Multi-entity |

### Deliverables
- `data/demo/golden_queries.jsonl` — curated queries with expected answers
- `scripts/run_demo_queries.py` — smoke test runner, flags regressions
- Run before every defense rehearsal

---

## Execution Order

```
Phase 2 (Refactor) → Phase 1 (Ablation) → Phase 3 (F1 fix) → Phase 4 (Demo)
```

Rationale: Refactoring first makes ablation trivial (toggle signals independently). Ablation reveals where F1 loss comes from. F1 fix is informed by data. Demo set is curated from queries that work well post-fix.

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Token F1 | 0.12 | > 0.40 |
| Context Hit Rate | 64.7% | > 70% (with best NER) |
| Demo pass rate | Unknown | 100% on golden set |
| agent.py lines | 911 | < 250 per file |
| hybrid.py lines | 702 | < 200 per file |

---

## Out of Scope
- Scaling to 5000+ pages (deferred to post-defense)
- DPO alignment (only if QLoRA alone doesn't hit 0.40)
- New NER backends beyond the existing 6
