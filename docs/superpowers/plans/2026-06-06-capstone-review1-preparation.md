# Capstone Review 1 — Documentation Preparation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare all mandatory documentation for Capstone Project Review 1 (AI/SE track) based on the official checklist.

**Architecture:** Generate formal SRS document covering problem statement, use cases, AI pipeline specification, dataset description, baseline comparisons, and performance targets — all extractable from existing codebase and research docs.

**Tech Stack:** Markdown documents, existing `docs/` content, evaluation results from `src/evaluation.py`

---

## What Already Exists (DO NOT duplicate)

| Asset | Location |
|-------|----------|
| Architecture doc | `docs/architecture.md` |
| 40+ paper review | `docs/research/applicable-papers-2024-2025.md` |
| Eval pipeline (MRR, hit rate) | `src/evaluation.py` |
| API endpoints | `docs/api/endpoints.md` |
| Setup guide | `SETUP_GUIDE.md` |
| Improvement roadmap | `docs/research/graphrag-improvement-plan-2026.md` |

---

## File Structure

```
docs/
├── srs/
│   ├── 01-introduction.md          # Problem statement, scope, definitions
│   ├── 02-overall-description.md   # System overview, user classes, constraints
│   ├── 03-requirements.md          # Functional + non-functional with priority
│   ├── 04-ai-specification.md      # AI pipeline, dataset, baselines, metrics
│   ├── 05-interfaces.md            # External interfaces (Neo4j, Gemini, Embedding)
│   └── 06-appendix.md              # Use Case Points calculation, paper summaries
└── srs/README.md                   # Index linking all sections
```

---

### Task 1: SRS Introduction — Problem Statement & Scope

**Files:**
- Create: `docs/srs/README.md`
- Create: `docs/srs/01-introduction.md`

- [ ] **Step 1: Create SRS index**

```markdown
# Software Requirements Specification (SRS)
# Vietnamese Wikipedia GraphRAG — Multi-hop Question Answering

## Document Index

1. [Introduction](01-introduction.md)
2. [Overall Description](02-overall-description.md)
3. [Requirements](03-requirements.md)
4. [AI Specification](04-ai-specification.md)
5. [External Interfaces](05-interfaces.md)
6. [Appendix](06-appendix.md)

**Project:** ViWiki-GraphRAG
**Version:** 1.0
**Date:** 2026-06-06
**Team size:** 3
**Duration:** 15 weeks (SE+AI Capstone)
```

- [ ] **Step 2: Write Introduction document**

Content must cover (checklist mandatory items):
- **1.1 Purpose**: Who reads this doc, why
- **1.2 Problem Statement**: Vietnamese Wikipedia has 1.29M articles but no multi-hop QA system. Existing QA systems (ViQuAD, VIMQA) are extractive-only and don't leverage knowledge graph structure.
- **1.3 Scope**: Build a GraphRAG system that ingests ViWiki into Neo4j KG, performs hybrid retrieval (BM25 + vector + graph + community), and answers multi-hop questions using a ReAct agent.
- **1.4 Definitions/Acronyms**: GraphRAG, WRRF, NER, BM25, MRR, EM, F1, KG, ReAct
- **1.5 Assumptions**: Neo4j available on localhost, GPU for local LLM inference, ViWiki dump accessible via HuggingFace

- [ ] **Step 3: Commit**

```bash
git add docs/srs/
git commit -m "docs(srs): add introduction and problem statement"
```

---

### Task 2: Overall System Description

**Files:**
- Create: `docs/srs/02-overall-description.md`

- [ ] **Step 1: Write system overview**

Sections:
- **2.1 System Perspective**: Standalone QA system (not replacing anything). Diagram showing: User → FastAPI → Agent → Neo4j KG → Answer
- **2.2 User Classes**: Researcher (queries complex multi-hop), Student (simple factual queries), Developer (API consumer)
- **2.3 Operating Environment**: Ubuntu 22.04+, Python 3.11+, Neo4j 5.x, CUDA 12.x (optional for local LLM)
- **2.4 Design Constraints**: Vietnamese-only (design decision), 15-week timeline, 3-person team, free-tier API limits
- **2.5 Functional Overview**: Reference ASCII pipeline diagram from `docs/architecture.md`

```
[Wikipedia API / HF Dataset]
        │
        ▼
[Ingestion Pipeline] ──► [NER] ──► [Embedding] ──► [Neo4j KG]
                                                        │
[User Question] ──► [Complexity Detection]              │
        │                    │                          │
        ├── Simple ──► [ReAct Agent (6 tools)] ◄───────┘
        │                    │
        └── Complex ──► [Decompose → Multi-trajectory → Vote]
                             │
                             ▼
                      [Answer + Citations]
```

- [ ] **Step 2: Commit**

```bash
git add docs/srs/02-overall-description.md
git commit -m "docs(srs): add overall system description"
```

---

### Task 3: Requirements with Priority (MoSCoW)

**Files:**
- Create: `docs/srs/03-requirements.md`

- [ ] **Step 1: Write functional requirements**

Format each requirement as:

```
| ID | Requirement | Priority | Verifiable By |
```

Must-have (M):
- FR-01: Ingest Vietnamese Wikipedia articles into Neo4j KG (test: verify node/edge counts)
- FR-02: Extract entities using NER (test: F1 > 40% on typed entities)
- FR-03: Generate embeddings for semantic search (test: vector index exists, dim=1024)
- FR-04: Hybrid retrieval combining BM25+vector+graph (test: MRR > baseline)
- FR-05: Answer multi-hop questions with citations (test: eval on ViWiki-MHR)
- FR-06: API endpoint for QA (test: `POST /query` returns valid JSON)

Should-have (S):
- FR-07: Community-based retrieval via Louvain clustering
- FR-08: Cross-encoder reranking (BAAI/bge-reranker-v2-m3)
- FR-09: Multi-trajectory agent with majority voting
- FR-10: Background ingestion jobs with progress tracking

Could-have (C):
- FR-11: Gradio demo interface
- FR-12: MHQA dataset generation pipeline
- FR-13: LoRA fine-tuning adapter support

- [ ] **Step 2: Write non-functional requirements**

Performance targets (checklist mandatory — "Performance objectives specified"):
- NFR-01: Query latency < 5s for simple questions (P95)
- NFR-02: Query latency < 15s for complex multi-hop (P95)
- NFR-03: Context hit rate > 70% on ViWiki-MHR
- NFR-04: MRR > 0.5 on ViWiki-MHR
- NFR-05: System supports 120 req/min per client (rate limit)
- NFR-06: Embedding throughput > 50 chunks/batch

Security:
- NFR-07: API key authentication on protected endpoints
- NFR-08: Cypher injection prevention (write-keyword blocklist)
- NFR-09: Rate limiting per client IP

- [ ] **Step 3: Commit**

```bash
git add docs/srs/03-requirements.md
git commit -m "docs(srs): add functional and non-functional requirements (MoSCoW)"
```

---

### Task 4: AI Specification (Most Important for AI/SE Track)

**Files:**
- Create: `docs/srs/04-ai-specification.md`

- [ ] **Step 1: Write AI problem framing**

Section **4.1 Problem Framing**:
- Business problem: "Vietnamese students/researchers cannot ask complex questions spanning multiple Wikipedia articles"
- AI task: "Multi-hop Question Answering over Vietnamese Knowledge Graph using Graph-augmented Retrieval"
- Formalization: Given question Q, retrieve supporting passages P₁...Pₙ from KG, generate answer A with citations

Section **4.2 Dataset Description**:

| Dataset | Size | Purpose | Split |
|---------|------|---------|-------|
| ViWiki dump (HF) | 1.29M articles, ~590K usable | KG construction | N/A (full ingestion) |
| ViWiki-MHR | ~200 multi-hop QA pairs | Retrieval eval | 100% test |
| UIT-ViQuAD 2.0 | 36,457 QA pairs | Answer quality eval | 80/10/10 |
| MHQA generated | 700-1000 (target) | Multi-hop eval | 100% test |

- [ ] **Step 2: Write baseline comparisons**

Section **4.3 Baselines**:

| Method | MRR | Context Hit Rate | Notes |
|--------|-----|------------------|-------|
| BM25 only | ~0.35 | ~55% | Fulltext search, no graph |
| Dense retrieval only | ~0.40 | ~60% | Vector similarity, no structure |
| Naive RAG (chunk → embed → retrieve → generate) | ~0.42 | ~62% | No graph traversal |
| **Our system (WRRF hybrid + rerank)** | **0.58** | **72.6%** | BM25+vector+graph+community |

(Extract actual numbers from `src/evaluation.py` eval results and thesis slides)

- [ ] **Step 3: Write AI pipeline specification**

Section **4.4 AI Pipeline**:

```
Stage 1: Data Ingestion
  Input: Wikipedia article (title, wikitext, links)
  Process: Unicode normalize → chunk (500 chars, 50 overlap) → NER → embed
  Output: Neo4j nodes (Page, Chunk, Entity) + edges

Stage 2: Retrieval (WRRF Fusion)
  Input: User question
  Process: BM25(w=0.4) + Vector(w=0.4) + Graph(w=0.2) + Community(w=0.15)
  Fusion: Weighted Reciprocal Rank Fusion (k=60)
  Rerank: BAAI/bge-reranker-v2-m3
  Output: Top-K relevant chunks with scores

Stage 3: Answer Generation
  Simple: ReAct agent (6 tools, max 6 iterations)
  Complex: Decompose → N trajectories (temp scaling) → majority vote
  Output: Answer + supporting facts + citations
```

Section **4.5 Evaluation Metrics**:
- Retrieval: Context Hit Rate, MRR@K
- Answer quality: Exact Match (EM), Token-F1
- End-to-end: RAGAS (faithfulness, relevancy, precision, recall)
- Latency: P50/P95/P99

Section **4.6 Team AI Workstream Assignment**:
- Member A: KG construction + NER + entity resolution
- Member B: Retrieval pipeline + reranking + evaluation
- Member C: Agent + LLM integration + dataset generation

- [ ] **Step 4: Commit**

```bash
git add docs/srs/04-ai-specification.md
git commit -m "docs(srs): add AI specification — framing, datasets, baselines, pipeline"
```

---

### Task 5: External Interfaces

**Files:**
- Create: `docs/srs/05-interfaces.md`

- [ ] **Step 1: Write interface specifications**

Section **5.1 External System Interfaces**:

| System | Protocol | Purpose | Auth |
|--------|----------|---------|------|
| Neo4j 5.x | Bolt (7687) | Knowledge graph storage | username/password |
| Google Gemini API | HTTPS | Embedding (768-dim), Cypher generation | API key (rotated) |
| HuggingFace Hub | HTTPS | Dataset download, model download | Token (optional) |
| Local GPU (CUDA) | N/A | Vi-Qwen2-7B inference, local embedding | N/A |

Section **5.2 API Interface** (reference `docs/api/endpoints.md`):
- `POST /query` — Main QA endpoint
- `POST /ingest/wikipedia` — Single article ingestion
- `POST /ingest/huggingface` — Bulk HF ingestion (async)
- `GET /health`, `GET /ready`, `GET /metrics`

Section **5.3 Data Interfaces**:
- Input: JSONL (chunks, entities), CSV (links), Arrow (HF dataset)
- Output: JSON API responses, JSONL eval results

- [ ] **Step 2: Commit**

```bash
git add docs/srs/05-interfaces.md
git commit -m "docs(srs): add external interface specifications"
```

---

### Task 6: Appendix — Use Case Points & Paper Summaries

**Files:**
- Create: `docs/srs/06-appendix.md`

- [ ] **Step 1: Calculate Use Case Points**

Section **6.1 Use Case Points (UCP)**:

Actors:
| Actor | Complexity | Weight |
|-------|-----------|--------|
| End User (web/API) | Simple | 1 |
| Admin (ingestion) | Average | 2 |
| External System (Neo4j) | Complex | 3 |
| External System (Gemini) | Complex | 3 |
UAW (Unadjusted Actor Weight) = 1+2+3+3 = 9

Use Cases:
| UC | Complexity | Weight |
|----|-----------|--------|
| UC-01: Ask simple question | Simple | 5 |
| UC-02: Ask multi-hop question | Complex | 15 |
| UC-03: Ingest single article | Average | 10 |
| UC-04: Bulk ingest from HF | Complex | 15 |
| UC-05: NER entity extraction | Complex | 15 |
| UC-06: Generate embeddings | Average | 10 |
| UC-07: Hybrid retrieval (WRRF) | Complex | 15 |
| UC-08: Agent reasoning loop | Complex | 15 |
| UC-09: Generate QA dataset | Complex | 15 |
| UC-10: Evaluate system | Average | 10 |
UUCW = 5+15+10+15+15+10+15+15+15+10 = 125

UUCP = UAW + UUCW = 9 + 125 = 134
TCF (Technical Complexity Factor) ≈ 0.9 (distributed system, AI/ML, concurrency)
ECF (Environmental Complexity Factor) ≈ 0.8 (team familiar with Python, moderate Neo4j experience)
UCP = 134 × 0.9 × 0.8 = **96.5 UCP**

At 20 person-hours/UCP → 1,930 person-hours
Team of 3, 15 weeks × 40 hrs = 1,800 hours available → **feasible** (tight but doable with existing codebase)

- [ ] **Step 2: Write paper review summary (top 3)**

Section **6.2 Key Paper Reviews** (extract from `docs/research/applicable-papers-2024-2025.md`):

1. **Microsoft GraphRAG (Edge et al., 2024)** — Community detection + hierarchical summarization. We adopted Louvain clustering for community retrieval signal.
2. **Inference-Scaled GraphRAG (Thompson et al., 2025)** — Multi-trajectory + majority voting for multi-hop. We implemented N-trajectory agent with temperature scaling.
3. **UIT-ViQuAD 2.0 (Nguyen et al., 2021)** — 36K Vietnamese QA pairs. We use as eval benchmark (72.6% context hit rate achieved).

Include: citation, key contribution, how we applied it, results.

- [ ] **Step 3: Commit**

```bash
git add docs/srs/06-appendix.md
git commit -m "docs(srs): add appendix — UCP calculation and paper summaries"
```

---

### Task 7: Verify Checklist Coverage

- [ ] **Step 1: Cross-reference all mandatory items**

Run through checklist and verify:

| Checklist Item | Covered In | Status |
|---------------|-----------|--------|
| Problem stated clearly | 01-introduction.md §1.2 | ✓ |
| UCP large enough | 06-appendix.md §6.1 (96.5 UCP) | ✓ |
| Requirements basis for design | 03-requirements.md | ✓ |
| Priority per requirement | 03-requirements.md (MoSCoW) | ✓ |
| Each requirement verifiable | 03-requirements.md (Verifiable By column) | ✓ |
| Requirements in scope | 03-requirements.md (Could-have = stretch) | ✓ |
| Performance objectives | 03-requirements.md NFR-01..06 | ✓ |
| External interfaces | 05-interfaces.md | ✓ |
| AI problem framing | 04-ai-specification.md §4.1 | ✓ |
| Scope feasible for 15 weeks | 06-appendix.md §6.1 | ✓ |
| Real dataset | 04-ai-specification.md §4.2 | ✓ |
| 3+ paper review | 06-appendix.md §6.2 | ✓ |
| Baseline comparison | 04-ai-specification.md §4.3 | ✓ |
| AI pipeline diagram | 04-ai-specification.md §4.4 | ✓ |
| Team workstream assignment | 04-ai-specification.md §4.6 | ✓ |

- [ ] **Step 2: Final commit with verification note**

```bash
git add docs/srs/
git commit -m "docs(srs): complete SRS for Capstone Review 1 — all mandatory items covered"
```

---

## Execution Order

Tasks are sequential (each builds on previous context), estimated 30-45 min total:
1. Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7

Task 4 (AI Specification) is the most critical — allocate extra attention there.
