# Project Specification — Vietnamese GraphRAG KGQA (ViWiki‑MHR)

## 1) Objective
Build a fully local, sovereignty‑preserving Vietnamese KGQA system that answers multi‑hop questions over Vietnamese Wikipedia using a typed Neo4j knowledge graph, a tool‑constrained SLM orchestrator, and strict citation‑based groundedness.

**Primary outcomes:**
- A working local QA service with deterministic tool calls and citations.
- An open, reproducible multi‑hop dataset (ViWiki‑MHR, ~36K).
- A fine‑tuned local SLM (Text2Cypher + tool‑calling).
- Evaluation suite with hallucination and unanswerable handling.

## 2) Scope
**In‑scope**
- Local single‑process pipeline (no external APIs).
- Vietnamese Wikipedia ingestion from the pinned `Keithsel/viwiki-20260523` snapshot, generated from raw MediaWiki XML.
- Typed KG schema + Cypher execution.
- Hybrid retrieval (graph + text fallback).
- Text2Cypher fine‑tuning with deterministic validation.
- Dataset generation + QC pipeline.

**Out‑of‑scope**
- Web search / external retrieval.
- Training or evaluation on gated datasets (e.g., ViMQA).
- Multi‑agent orchestration (single‑agent ReAct only).

## 3) System Architecture (High‑Level)
**Flow:** User query → SLM orchestrator → ReAct loop (≤6 steps) → tools → citation verifier → final answer.

**Tools (strictly 4):**
1. `kg_schema()` → JSON schema (cached).
2. `kg_query(cypher)` → Neo4j execution results or compiler errors.
3. `text_search(query, k)` → hybrid BM25 + dense retrieval.
4. `get_passage(passage_id)` → raw paragraph text.

**Storage:**
- **Neo4j** for KG (typed entities/relations, provenance).
- **Qdrant/FAISS** + **BM25** for fallback text retrieval.
- Processed Vietnamese Wikipedia Parquet corpus (`Keithsel/viwiki-20260523`) as the passage source.

## 4) Knowledge Graph Design
**Node types:** `Person`, `Place`, `Organization`, `Event`, `TácPhẩm`, plus `Chunk`/`Page` for provenance.

**Relations (examples):** `:SÁNG_TÁC`, `:GIA_NHẬP`, `:TRỤ_SỞ`, `:BỐ_CẢNH`, `:SINH_NĂM`.

**Provenance:**
- Every entity/edge must link to passages and page IDs.
- Passage‑level IDs used for citation validation.

## 5) Text2Cypher Pipeline
**Stages:**
1. **Schema linking:** BM25/TF‑IDF over labels/relations/properties to prune schema.
2. **SLM generation:** produce Cypher from query + pruned schema.
3. **Deterministic validation:** syntax/schema/security checks; block destructive clauses.
4. **Error refinement:** compiler errors fed back into the ReAct loop for correction.

**Goal:** >95% executable Cypher without external API calls.

## 6) Orchestration & Groundedness
**ReAct loop:** capped at 6 iterations to avoid runaway loops.
**Sufficiency checks:** if KG evidence empty/inconsistent → route to `text_search` or abstain.
**Citation verification:** final answer must cite `passage_id`s present in retrieval history.

## 7) Dataset: ViWiki‑MHR
**Size:** ~36K examples, open license (CC‑BY‑SA).
**Sources:** UIT‑ViQuAD 2.0 + synthetic multi‑hop KG walks grounded in the raw XML-derived Vietnamese Wikipedia snapshot published at `Keithsel/viwiki-20260523`.

**Corpus preparation:** `scripts/viwiki_processing/` streams the 2026-05-23 Vietnamese Wikipedia MediaWiki XML dump, filters main-namespace articles, cleans wikitext into plain text, and exports both cleaned and raw Parquet shards. The Hugging Face dataset repo is the reproducible distribution point for downstream ingestion, QA generation, and evaluation.

**Schema fields:**
- `question`, `answer`, `num_hops`, `reasoning_type`, `answerable`.
- `gold_passage_ids`, `cypher_query`, `decomposition_annotations`.

**Reasoning taxonomy:** lookup, bridge, comparison, intersection, temporal, fan‑out.

**Adversarial unanswerables:** broken‑link multi‑hop generation (BRINK‑style).

**QC pipeline:** grounding match → NLI entailment → LLM coherence → human spot‑check.

## 8) Model Strategy
**Base model:** Sailor2‑8B or Qwen2.5‑7B.

**Training:**
- **QLoRA** (NF4, double quantization, LoRA on all linear layers).
- **DPO** alignment for JSON tool‑call compliance and schema correctness.

**Hardware target:** 16GB RAM, 8GB VRAM (RTX 3060/4060 or Apple Silicon).

## 9) Evaluation
**Metrics (RAG Triad):** context relevance, faithfulness, answer relevance.
**Hallucination taxonomy:** intrinsic/extrinsic (ViHallu).
**Robustness:** incomplete KG tests (BRINK‑style).
**Baselines:** vector‑only, BM25‑only, graph‑only, hybrid.

## 10) Deliverables
1. Local KGQA service with 4 tools and citation verifier.
2. ViWiki‑MHR dataset release + documentation.
3. Fine‑tuned Text2Cypher SLM + adapters.
4. Evaluation report with tables/plots.

## 11) Risks & Mitigations
- **Vietnamese tokenization errors:** use underthesea/pyvi preprocessing; NER pipeline.
- **KG sparsity:** hybrid retrieval fallback + sufficiency checks.
- **Hallucinations:** deterministic verification + abstain policy.
- **Compute limits:** QLoRA/DPO + strict iteration cap.

## 12) Milestones
1. **Base SLM local** (4‑bit running).
2. **End‑to‑end baseline** (KG + tools + ReAct on 100 queries).
3. **Text2Cypher adapter** (QLoRA gains vs base).
4. **Full ViWiki‑MHR generation** + QC.
5. **Evaluation** + report.
