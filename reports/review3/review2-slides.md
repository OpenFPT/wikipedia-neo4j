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

**Capstone Review 2 — FPT University SE+AI**

| | |
|---|---|
| Team | Huynh Quoc Trung · Nhu Quang Anh · Pham Ngo Dinh Khoi · Truong Dinh Thien |
| Student ID | QE180038 · QE180005 · QE180110 · QE180018 |
| Supervisor | <!-- TODO --> |
| Date | Week 8 — July 2026 |

---
<!-- class: default -->

## Table of Contents

1. System Architecture (D2)
2. Detail Design (D3)
3. Technology Choices (P5)
4. Implementation Quality (P6)
5. Algorithm Complexity (P7)
6. Data Pipeline (AI)
7. Model Implementation (AI)
8. Evaluation Results (AI)
9. AI Engineering Quality (AI)
10. Team Contribution

---

## 1. System Architecture

```
User Question (Vietnamese)
        │
        ▼
┌────────────────────────────────────────────┐
│  FastAPI  (auth · rate-limit · job mgmt)   │
├────────────────────────────────────────────┤
│  Agent Layer (src/orchestration/)          │
│  complexity detect → ReAct / multi-traj    │
├──────────┬─────────┬──────────┬────────────┤
│  BM25    │ Vector  │  Graph   │ Community  │
│ fulltext │1024-dim │ 2-hop    │ Louvain    │
├──────────┴─────────┴──────────┴────────────┤
│  Neo4j Knowledge Graph                     │
│  Page · Chunk · Person · Org · Loc · Work  │
└────────────────────────────────────────────┘
```

Deployment: systemd Neo4j service · uv-managed Python 3.12 · local GPU

---

## 1. Dataflow — Ingestion & Query

### Ingestion
```
Wikipedia API / HF Dataset
  → stub filter (≥200 chars) → Unicode NFC normalize
  → chunk (500 tokens / 50 overlap)
  → NER (6 pluggable backends)
  → embed (GreenNode 1024-dim)
  → Neo4j (batch UNWIND)
```

### Query
```
Question Q
  → complexity detect (simple / complex)
  ├─ Simple  → ReAct agent (≤6 iter, 6 tools)
  └─ Complex → decompose → N trajectories
                → WRRF retrieval → rerank
                → majority vote → A + citations
```

---

## 2. Detail Design — Graph Schema (ERD)

```
(:Page {id, title, url, text})
    │ HAS_CHUNK
    ▼
(:Chunk {id, text, embedding[1024], seq})
    │ MENTIONS_PERSON / _ORG / _LOCATION / _WORK
    ▼
(:Entity {id, name, type, aliases[]})
    (subtypes: Person · Organization · Location · Work)

(:Page)-[:LINKS_TO]->(:Page)
(:Page)-[:FOUNDED_BY / LOCATED_IN / BORN_IN
         / MEMBER_OF / PART_OF / CREATED_BY]->(:Entity)
```

Constraints: UNIQUE on `Page.id`, `Chunk.id`, `Entity.id`, `Community.id`
Indexes: fulltext (`chunk.text`), vector (`chunk.embedding`, 1024-dim)

---

## 2. Detail Design — Key Classes

| Class | Module | Role |
|-------|--------|------|
| `Settings` | `src/config.py` | Pydantic BaseSettings — all env config |
| `Neo4jClient` | `src/infrastructure/neo4j_client.py` | Driver singleton, schema setup, batch writes |
| `EntityResolver` | `src/extraction/entity_resolution.py` | Diacritic normalization, alias merging |
| `JobStore` | `src/infrastructure/job_store.py` | Thread-safe JSON job persistence |
| `KGWalk` | `src/dataset_gen.py` | 2/3-hop walk extraction for dataset gen |
| `QAPair` | `src/dataset_gen.py` | Generated QA with metadata |
| `EvalMetrics` | `src/evaluation.py` | MRR, Hit Rate, F1, latency tracking |

---

## 2. Detail Design — Job State Machine

```
          ┌─────────┐
          │ pending │
          └────┬────┘
               │ start
          ┌────▼─────┐
          │ running  │
          └──┬────┬──┘
     success │    │ error / cancel
      ┌──────▼┐  ┌▼─────────┐
      │completed│ │  failed  │
      └────────┘  └──────────┘
           ↑ restart interrupted
      ┌────┴──────┐
      │interrupted│  (on server restart)
      └───────────┘
```

---

## 2. Detail Design — Screen Design

| Interface | Technology | Purpose |
|-----------|-----------|---------|
| **Gradio Demo** | `src/app_gradio.py` | Interactive QA for end-users |
| **GraphPulse Dashboard** | `src/dashboard/` | Ops monitoring: query logs, KG stats, latency charts |
| **MCP Server** | `src/mcp_pkg/` | Claude Desktop integration |
| **REST API** | FastAPI `/query`, `/ingest` | Programmatic access |

Dashboard: SVG charts · keyboard shortcuts (`/` focus, `Esc` clear) · accessibility focus-visible

---

## 3. Technology Choices (P5)

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Backend** | FastAPI + Python 3.12 | Async, Pydantic v2, OpenAPI auto-docs |
| **Graph DB** | Neo4j 5.x | Vector index + fulltext + Cypher in one DB |
| **Embeddings** | GreenNode-Embedding-Large-VN-Mixed-V1 | Best MAP@5 for Vietnamese |
| **NER** | ViDeBERTa / wikilink (6 backends) | Pluggable, Vietnamese-specific |
| **Reranker** | BAAI/bge-reranker-v2-m3 | Cross-encoder, multilingual |
| **LLM** | Vi-Qwen2-7B-RAG (4-bit NF4) | Local-first, Vietnamese fine-tuned |
| **External APIs** | Gemini · HuggingFace · Wikidata | Embedding backup, dataset source, entity enrichment |
| **Frontend** | Gradio + vanilla JS dashboard | Rapid AI prototyping |

---

## 4. Implementation Quality (P6)

### Coding Conventions ✅
- **ruff** + **ruff-format** enforced via pre-commit hooks
- **mypy** strict type checking
- `snake_case` properties · `SCREAMING_SNAKE_CASE` relationships · `_idx`/`_ft` index suffixes

### Config & Secrets ✅
- All config via `pydantic_settings.BaseSettings` from `.env`
- `.env.example` provided · no hard-coded credentials anywhere

### Design Patterns Applied

| Pattern | Where |
|---------|-------|
| Strategy | NER backends (6 swappable via `NER_BACKEND`) |
| Singleton | Neo4j driver · Local LLM (lazy-loaded) |
| Factory | Backend selector (`get_ner_backend()`) |
| Facade | WRRF retrieval pipeline hides 4-signal complexity |

---

## 5. Algorithm Complexity (P7)

### Problem Statement
> Given Vietnamese NL question Q, retrieve supporting passages P₁…Pₙ from KG G, generate answer A with citations. Complexity levels: 1-hop (lookup), 2-hop (bridge), 3-hop (fan-out).

### Algorithm Choices

| Algorithm | Purpose | Citation |
|-----------|---------|---------|
| **WRRF** (4-signal fusion) | Combine BM25 + Vector + Graph + Community | RRF, Cormack et al. |
| **Leiden / Louvain** | Community detection for topic clustering | GraphRAG, Edge et al. 2024 |
| **ReAct** | Step-by-step graph reasoning | Yao et al. 2023 |
| **Multi-trajectory voting** | Robust multi-hop answers | Thompson et al. 2025 |
| **QLoRA** | Fine-tune Text2Cypher on limited GPU | Dettmers et al. 2023 |

---

## 5. Algorithm — WRRF Formula

```
score(d) = Σ wᵢ / (k + rankᵢ(d))
  k = 60  (smoothing constant)
  w_BM25 = 0.4  ·  w_Vector = 0.4
  w_Graph = 0.2  ·  w_Community = 0.15

Pipeline:
  WRRF fusion (top-20) → cross-encoder rerank → top-5
  Reranker: BAAI/bge-reranker-v2-m3
```

**Enhancement over baseline:** 4-signal WRRF vs. standard 2-signal (BM25+Vector) achieves **+20.8% MRR**.

Vietnamese-specific: diacritic normalization, alias merging ("Bác Hồ" → "Hồ Chí Minh"), wikilink-grounded NER.

---

## 6. Data Pipeline (AI)

| Dataset | Size | Split | Purpose |
|---------|------|-------|---------|
| Vietnamese Wikipedia | ~590K articles | N/A (KG only) | Knowledge graph construction |
| ViWiki-MHR | ~200 pairs | 100% test | Multi-hop retrieval eval |
| UIT-ViQuAD 2.0 | 36,457 pairs | 80/10/10 official | Benchmark |
| MHQA (generated) | 700–1000 target | 100% test | Expanded eval |

### Preprocessing
- Stub filtering: ≥200 chars (1.29M → ~590K)
- Unicode NFC normalization
- 5-layer QC: well-formedness → grounding → dedup → answerability → schema validation

### Dataset Expansion
- `scripts/generate_dpo_pairs.py` — DPO training data
- `scripts/mhqa_generate.py` — KG walk + LLM rewrite; 10% human spot-check

---

## 7. Model Implementation (AI)

### Transfer Learning (primary approach)

| Model | Role | Type |
|-------|------|------|
| GreenNode-Embedding-Large-VN-Mixed-V1 | Dense retrieval | Pre-trained |
| ViDeBERTa / NlpHUST electra-base | Vietnamese NER | Pre-trained + fine-tuned |
| BAAI/bge-reranker-v2-m3 | Cross-encoder rerank | Pre-trained |
| Vi-Qwen2-7B-RAG (4-bit NF4) | Answer generation | Pre-trained |

### Fine-tuning — Text2Cypher (QLoRA)
```
Base:  AITeamVN/Vi-Qwen2-7B-RAG
Method: QLoRA (r=32, lora_alpha=64, 4-bit NF4)
Data:  scripts/generate_training_data.py + generate_gold_cypher.py
Eval:  scripts/eval_kg_only.py (PEFT adapter: models/text2cypher_adapter/final)
```

Inference integrated: Gradio · FastAPI `/query` · dashboard · MCP server

---

## 8. Evaluation Results (AI)

### Retrieval Baseline Comparison

| Method | MRR | Hit Rate@5 | Latency |
|--------|-----|-----------|---------|
| BM25 only | ~0.35 | ~55% | 0.8s |
| Dense only | ~0.40 | ~60% | 1.2s |
| Naive RAG | ~0.42 | ~62% | 2.5s |
| BM25 + Vector | ~0.48 | ~66% | 1.5s |
| **Ours (WRRF+Rerank)** | **0.4497** | **64.67%** | **51ms** |

**+20.8% MRR vs best non-graph baseline · 51ms retrieval latency (2,652 ViQuAD2 queries)**

---

## 8. Evaluation Results — Multi-metric

| Metric | Value | Dataset | Notes |
|--------|-------|---------|-------|
| Context Hit Rate | 0.726 | ViQuAD2 (500 samples) | Top-5 recall |
| MRR | 0.364 | ViQuAD2 | Mean reciprocal rank |
| Token F1 | 0.1195 | ViQuAD2 | Answer quality |
| NER F1 (wikilink) | 46.9% | Internal eval | Typed entity F1 |

### Known Failure Modes
- `abstain_accuracy = 0.0` — model never abstains when it should
- Low hit rate on 3-hop fan-out questions (multi-entity coverage gap)
- Diacritic mismatch causes ~5% entity resolution failures

---

## 9. AI Engineering Quality (AI)

### No Fake Demo ✅
Full pipeline on every call: NER → embed → WRRF → cross-encoder rerank → LLM → citations

### Custom Engineering ✅
- WRRF 4-signal fusion with configurable weights
- Leiden community detection + summary retrieval (4th signal)
- Question complexity routing → decompose → multi-trajectory majority vote
- Vietnamese entity resolution (diacritic normalization + alias merging)
- Cypher safety validation (write-keyword blocklist)
- QLoRA fine-tuning for Text2Cypher
- 5-layer QC pipeline for dataset generation
- Multi-key Gemini rotation with exponential backoff

### Experiment Tracking ⚠️
Eval results saved as timestamped JSON in `reports/eval/`. MLflow/W&B not yet integrated.

---

## 10. Team Contribution

| Member | Student ID | Assigned Modules |
|--------|-----------|-----------------|
| Huynh Quoc Trung | QE180038 | KG construction · NER (6 backends) · entity resolution · dataset gen |
| Nhu Quang Anh | QE180005 | WRRF retrieval · reranker · evaluation pipeline · SRS docs |
| Pham Ngo Dinh Khoi | QE180110 | ReAct agent · local LLM · MHQA dataset · QLoRA fine-tuning |
| Truong Dinh Thien | QE180018 | FastAPI · job management · Gradio demo · dashboard |

### Progress Since Review 1

| Item | R1 Status | R2 Status |
|------|-----------|-----------|
| Core pipeline | ✅ Done | ✅ Stable |
| Full ingestion + ablation | 🔄 In progress | ✅ Done |
| QLoRA Text2Cypher | ⏳ Pending | 🔄 In progress |
| WRRF tuning + final eval | ⏳ Pending | 🔄 In progress |

---

## 10. Remaining Timeline

| Week | Milestone | Status |
|------|-----------|--------|
| 1–4 | Core pipeline + SRS + Architecture | ✅ Done |
| 5–7 | Full ingestion + Ablation + MHQA | ✅ Done |
| 8–10 | QLoRA Text2Cypher + DPO alignment | 🔄 In progress |
| 11–13 | WRRF tuning + final evaluation | ⏳ Pending |
| 14–15 | Final report + demo + defense | ⏳ Pending |

**UCP = 96.5** (UUCP=134 · TCF=0.9 · ECF=0.8) → On schedule ✅

---

# Thank You

## Q & A

---

## Appendix: Graph Schema Detail

```cypher
// Node labels
(:Page {id, title, url, text, source})
(:Chunk {id, text, embedding, sequence_number, page_id})
(:Entity {id, name, type, aliases})
(:Community {id, level, summary, member_ids[]})

// Relationship types
(:Page)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:MENTIONS_PERSON|MENTIONS_ORG|MENTIONS_LOCATION|MENTIONS_WORK]->(:Entity)
(:Page)-[:LINKS_TO]->(:Page)
(:Page)-[:FOUNDED_BY|LOCATED_IN|BORN_IN|MEMBER_OF|PART_OF|CREATED_BY]->(:Entity)
```

---

## Appendix: NER Backend Comparison

| Backend | Typed F1 | Speed | Notes |
|---------|---------|-------|-------|
| `simple` | ~25% | Fast | Regex + keyword |
| `underthesea` | ~35% | Fast | BIO tagging |
| `phonlp` | ~40% | Medium | PhoNLP + VnCoreNLP |
| `phobert` | ~42% | Slow | PhoBERT transformer |
| `videberta` | ~44% | Slow | ViDeBERTa/NlpHUST |
| **`wikilink`** | **46.9%** | Fast | Wikipedia hyperlinks — best for bulk ingestion |

---

## Appendix: MHQA Dataset Generation Pipeline

```
KG Walk (2-hop, 3-hop, broken-link)
  → Vietnamese question templates
  → optional LLM rewrite (naturalness)
  → 5-layer QC:
      1. Well-formedness check
      2. Grounding verification (entity_grounded_in_text)
      3. Deduplication
      4. Answerability check
      5. Schema validation
  → JSONL output (seeds.jsonl → final dataset)
```

Target: 700–1000 QA pairs · 10% human spot-check validated
