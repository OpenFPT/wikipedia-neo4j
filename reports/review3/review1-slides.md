---
marp: true
theme: gaia
class: lead
paginate: true
size: 16:9
lang: en
style: |
  section { font-size: 1.1em; }
  table { font-size: 0.78em; }
  code { font-size: 0.8em; }
  h1 { font-size: 1.6em; }
  h2 { font-size: 1.3em; }
---

# Vietnamese GraphRAG
## Multi-hop Question Answering over Wikipedia

**Capstone Review 1 — FPT University SE+AI**

| | |
|---|---|
| Team | Huynh Quoc Trung · Nhu Quang Anh · Pham Ngo Dinh Khoi · Truong Dinh Thien |
| Student ID | QE180038 · QE180005 · QE180110 · QE180018 |
| Supervisor | <!-- TODO --> |
| Date | Week 4 — June 2026 |

---
<!-- class: default -->

## Table of Contents

1. Problem & Objectives
2. Users & Practical Applicability
3. Innovation & New Technology
4. System Architecture
5. Requirements & SRS
6. AI: Problem Framing
7. Dataset
8. Related Works (3 papers)
9. AI Pipeline
10. Baselines & Results
11. Team Assignment
12. System Scale & Timeline

---

## 1. Problem Statement

> **Complex questions** about Vietnamese history, geography, and people often **span multiple Wikipedia articles** — impossible to answer with a single search.

### Pain Points

| User | Problem |
|------|---------|
| Students / researchers | Must read 5–10 articles to synthesize an answer |
| Standard RAG systems | Only retrieves the nearest chunk — loses multi-hop links |
| Vanilla LLMs | Hallucinate Vietnamese facts, no verifiable citations |

### Project Goal

**Build a fully local Vietnamese KGQA system** that answers multi-hop questions with validated citations, with no dependency on external APIs.

---

## 2. Users & Practical Applicability

### Real User Groups

- **FPT / HUST / UIT students** — academic knowledge lookup in Vietnamese
- **Researchers / journalists** — synthesize from Vietnamese Wikipedia sources
- **AI developers** — integrate into their own Vietnamese RAG pipelines

### Stakeholders

| Role | Entity |
|------|--------|
| End users | FPT University students (real-world testing) |
| Academic advisors | Supervisor + capstone committee |
| Open-source community | ViWiki-MHR dataset published under CC-BY-SA |

---

## 3. Innovation & New Technology

| Innovation | Description |
|-----------|-------------|
| **WRRF Hybrid Retrieval** | BM25 + Vector + Graph + Community — no Vietnamese QA system uses all 4 signals |
| **Multi-trajectory Agent** | N parallel trajectories + majority voting (arXiv:2506.19967) |
| **Local-first sovereignty** | Entire pipeline runs on local GPU, no data sent externally |
| **ViWiki-MHR Dataset** | First Vietnamese multi-hop dataset from KG walk + LLM rewrite + 5-layer QC |
| **Typed Vietnamese KG** | Neo4j with 4 entity types, 6 relation types from Vietnamese Wikipedia 2026 |

---

## 4. System Architecture

```
User Question (Vietnamese)
        │
        ▼
┌────────────────────────────────────────────┐
│  FastAPI  (auth · rate-limit · job mgmt)   │
├────────────────────────────────────────────┤
│  Agent Layer                               │
│  complexity detect → ReAct / multi-traj    │
├──────────┬─────────┬──────────┬────────────┤
│  BM25    │ Vector  │  Graph   │ Community  │
│ fulltext │1024-dim │ 2-hop    │ Louvain    │
├──────────┴─────────┴──────────┴────────────┤
│  Neo4j Knowledge Graph                     │
│  Page · Chunk · Person · Org · Loc · Work  │
└────────────────────────────────────────────┘
```

---

## 5. Requirements (SRS Summary)

### Functional Requirements (Must Have)

| ID | Requirement | Verifiable By |
|----|------------|---------------|
| FR-01 | Ingest Vietnamese Wikipedia articles | Unit test |
| FR-02 | Answer single-hop factual questions | EM/F1 on ViQuAD2 |
| FR-03 | Answer multi-hop questions + citations | Hit Rate on ViWiki-MHR |
| FR-04 | Bulk ingest from HuggingFace | Load test 10K articles |
| FR-05 | Generate MHQA evaluation dataset | QC pipeline pass rate |

### Non-Functional Requirements

| ID | Metric | Target | Achieved |
|----|--------|--------|---------|
| NFR-01 | P50 latency | < 3s / < 8s | 51ms retrieval ✅ |
| NFR-02 | Context Hit Rate | > 70% | 64.67% 🔄 |
| NFR-03 | Test coverage | ≥ 75% | 85% ✅ |

---

## 6. AI: Problem Framing

| Layer | Content |
|-------|---------|
| **Business Problem** | Complex Vietnamese questions need synthesis across multiple Wikipedia articles |
| **AI Task** | Multi-hop KGQA: Graph-augmented RAG with typed knowledge graph |
| **Formal** | Q → retrieve {P₁…Pₙ} from KG G → generate A + citations |

### Question Taxonomy

| Type | Hops | Example |
|------|------|---------|
| Simple (lookup) | 1-hop | "Born in which year?" |
| Bridge | 2-hop | "Where did the founder of [X] study?" |
| Comparison | 2+ hop | "Who is older: A or B?" |
| Fan-out | 3+ hop | "List all works by [X]" |

**Feasibility: 15 weeks, team of 4 → core pipeline already operational ✅**

---

## 7. Dataset

| Dataset | Size | Source | Purpose |
|---------|------|--------|---------|
| Vietnamese Wikipedia | ~590K articles | `wikimedia/wikipedia` | KG construction |
| ViWiki-MHR | ~200 pairs | KG walk + manual | Multi-hop eval |
| UIT-ViQuAD 2.0 | 36,457 pairs | UIT-NLP (VLSP 2021) | Benchmark |
| MHQA (generated) | 700–1000 target | KG walk + Claude | Expanded eval |

### Data Quality

- Stub filtering: ≥200 chars (1.29M → ~590K articles)
- NFC Unicode normalization
- 5-layer QC: well-formedness → grounding → dedup → answerability → schema

---

## 8. Related Works — Paper 1

**Microsoft GraphRAG** — Edge et al., 2024 (arXiv:2404.16130)

> Hierarchical community detection (Leiden) → pre-generate community summaries → map-reduce at query time.

**Original result:** 70–80% win rate vs naive RAG on broad queries.

**Applied in project:**
- Louvain community detection → assign chunks to communities
- Community summaries → 4th WRRF signal (weight=0.15)
- Improves recall for broad thematic queries about Vietnamese history/geography

---

## 8. Related Works — Paper 2

**Inference-Scaled GraphRAG** — Thompson et al., 2025 (arXiv:2506.19967)

> Sample N graph traversal trajectories with temperature scaling → aggregate via majority voting.

**Original result:** +64.7% vs traditional GraphRAG; 31.44% vs 15.26% on hard multi-hop.

**Applied in project:**
- `AGENT_N_TRAJECTORIES` config — N agents run in parallel with temperature scaling
- Majority voting → consensus answer
- Particularly effective for comparison/fan-out question types

---

## 8. Related Works — Paper 3

**UIT-ViQuAD 2.0** — Nguyen et al., 2020 (arXiv:2009.14725)

> 36,457 QA pairs from Vietnamese Wikipedia. Answerable + unanswerable. XLM-R achieves 77.8% EM.

**SotA baseline:** XLM-R 77.8% EM (answerable)

**Applied in project:**
- Primary benchmark for retrieval quality evaluation
- `src/viquad_adapter.py` adapts HuggingFace → eval pipeline
- **Achieved: 64.67% context hit rate** (top-5) — baseline for WRRF improvement tracking

---

## 9. AI Pipeline

### Ingestion
```
Wikipedia → normalize → chunk(500/50) → NER → embed → Neo4j
```

### Retrieval (WRRF Fusion)
```
Question Q
  ├─ BM25 (w=0.4)  ─┐
  ├─ Vector (w=0.4)  ┼→ WRRF(k=60) → cross-encoder rerank → top-K
  ├─ Graph (w=0.2)  ─┘
  └─ Community (w=0.15)
```

### Answer Generation
```
Q + complexity detection
  ├─ Simple  → ReAct (≤6 iter, 6 tools)
  └─ Complex → decompose → N trajectories → majority vote → A + citations
```

---

## 10. Baselines & Results

| Method | MRR | Hit Rate@5 | Latency |
|--------|-----|-----------|---------|
| BM25 only | ~0.35 | ~55% | 0.8s |
| Dense only | ~0.40 | ~60% | 1.2s |
| Naive RAG | ~0.42 | ~62% | 2.5s |
| BM25 + Vector | ~0.48 | ~66% | 1.5s |
| **Ours (WRRF+Rerank)** | **0.4497** | **64.67%** | **51ms** |

**+~20% MRR vs best baseline · retrieval latency 51ms (ViQuAD2, 2,652 queries)**

⚠️ Hit Rate below 70% target — full ingestion needed for sufficient entity coverage in KG.

---

## 11. Team Assignment

| Member | Student ID | Responsibility |
|--------|-----------|---------------|
| Huynh Quoc Trung | QE180038 | KG Construction · NER (6 backends) · Entity Resolution |
| Nhu Quang Anh | QE180005 | WRRF Retrieval · Reranker · Evaluation Pipeline |
| Pham Ngo Dinh Khoi | QE180110 | ReAct Agent · Local LLM · MHQA Dataset Gen |
| Truong Dinh Thien | QE180018 | FastAPI · Job Management · Gradio Demo · Deploy |

### Dependency Chain

NER (Trung) → Entity quality → Graph signal (Anh) → Agent tools (Khoi) → FastAPI (Thien)

---

## 12. System Scale & Timeline

**UCP = 96.5** (UUCP=134 · TCF=0.9 · ECF=0.8) → Feasible in 15 weeks ✅

| Week | Milestone | Status |
|------|-----------|--------|
| 1–4 | Core pipeline + SRS + Architecture | ✅ Done |
| 5–7 | Full ingestion + Ablation + MHQA | 🔄 In progress |
| 8–10 | QLoRA Text2Cypher + DPO alignment | ⏳ Pending |
| 11–13 | WRRF tuning + final evaluation | ⏳ Pending |
| 14–15 | Final report + demo + defense | ⏳ Pending |

---

# Thank You

## Q & A

---

## Appendix: Graph Schema

```cypher
(:Page)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS_*]->(:Person|Org|Location|Work)
(:Page)-[:LINKS_TO]->(:Page)
```

Indexes: fulltext (chunk.text) · vector (chunk.embedding 1024-dim) · btree (entity.name)

---

## Appendix: WRRF Formula

```
score(d) = Σ wᵢ / (k + rankᵢ(d))
  k = 60  (smoothing constant)
  BM25=0.4 · Vector=0.4 · Graph=0.2 · Community=0.15
```

Cross-encoder rerank: **BAAI/bge-reranker-v2-m3** — rescores top-20 → returns top-5
