# Project Gap Analysis and Rebuilding Strategy

This document establishes the strategic delta between our group's active English/API-based baseline demo and our final thesis vision (a fully local, sovereignty-focused Vietnamese Agentic GraphRAG system with a customized Vietnamese multi-hop dataset and local fine-tuned SLM orchestrator).

---

## 1. Executive Summary & Coverage Mapping

A comprehensive code audit reveals that our current FastAPI-based baseline represents roughly **10–15% of the target system architecture** and **0% of the dataset/evaluation workstream**. 

```
                                  [Target System: 100%]
+-----------------------------------------------------------------------------------+
| [X] FastAPI Skeleton, Logging, Docker, Pre-commit (10-15% - Operational Shell)     |
| [ ] Vietnamese Language Processing & Tokenization (0% - Greenfield)               |
| [ ] Typed Relational Knowledge Graph Schema (0% - Greenfield)                     |
| [ ] Local SLM Stack & Quantization Wrapper (0% - Greenfield)                       |
| [ ] ReAct Agentic Iteration Loop (0% - Greenfield)                                |
| [ ] ViWiki-MHR Dataset Generation & Verification Pipeline (0% - Greenfield)      |
| [ ] QLoRA Fine-tuning & Adapter Alignment Pipeline (0% - Greenfield)              |
| [ ] Multi-hop Evaluation & Groundedness Harness (0% - Greenfield)                 |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Gap Analysis

The current code serves as an operational scaffold, but its core functional pipelines are fundamentally decoupled from our target thesis objectives:

### 1. The Language Gap (Critical Failure Point)
*   **Current State:** Configured entirely for English Wikipedia (`20231101.en`). The baseline entity extractor relies on a naive English Title-Case regex:
    ```python
    entity_regex = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
    ```
*   **The Delta:** This regex will completely fail on Vietnamese. In Vietnamese, common nouns and proper nouns capitalize differently, spaces separate syllables rather than semantic words, and diacritics break basic character class boundaries.
*   **The Rebuild:** Integrate a specialized Vietnamese segmenter (e.g., `underthesea` or `pyvi`) and replace the regex with a robust, Vietnamese-aware Named Entity Recognition (NER) pipeline or LLM-assisted entity extraction script.

### 2. The Knowledge Graph Gap (Minimal Relational Semantics)
*   **Current State:** Implements a flat structural schema (`Page` $\xrightarrow{\text{HAS\_CHUNK}}$ `Chunk` $\xrightarrow{\text{MENTIONS}}$ `Entity`). This is not a semantic Knowledge Graph; it is a vector store index with text chunks mapped to unresolved entity strings.
*   **The Delta:** To perform multi-hop reasoning (e.g., matching authors to dates, and dates to birthplaces), the graph must have typed entities and semantically rich relationships.
*   **The Rebuild:** Extend the Neo4j schema to include:
    *   *Typed Nodes:* `Person`, `Place`, `Organization`, `Event`, `TácPhẩm`.
    *   *Typed Directed Relations:* `[:GIA_NHẬP]`, `[:SÁNG_TÁC]`, `[:TRỤ_SỞ]`, `[:BỐ_CẢNH]`, `[:SINH_NĂM]`.
    *   *Temporal/Numerical Attributes:* Storing years, populations, and active epochs as indexed node properties.

### 3. The Agent & Orchestration Gap (One-Shot Retrieval)
*   **Current State:** The retrieval function `query_graph()` is a single-shot execution. It queries Gemini to generate a Cypher script, tries it, and if it fails, falls back to full-text search. The "answer" is formed by a deterministic string concatenation of snippets:
    ```python
    answer = "Deterministic demo answer from retrieved graph context: " + " | ".join(snippets[:2])
    ```
*   **The Delta:** No logical reflection, no tool selection, no multi-hop question decomposition, and no true answer synthesis with citation tracking.
*   **The Rebuild:** Replace the single-shot routing with an active **ReAct Agent Loop** (*Thought $\rightarrow$ Action $\rightarrow$ Observation*). The model must write JSON actions to call `kg_query`, `text_search`, or `get_passage`, process errors natively, and generate structured, cited final responses.

### 4. The Local SLM Gap (Google Gemini Dependency)
*   **Current State:** The backend calls Google's hosted Gemini API exclusively for embeddings, Cypher generation, and text queries. Every run is dependent on active internet access and external token billing.
*   **The Delta:** This violates our core sovereignty design (100% private, local deployment).
*   **The Rebuild:** Swap the Gemini client wrappers for a local SLM integration stack (using `vLLM`, `llama.cpp`, or `Transformers` running a quantized 7B model in 4-bit NF4) paired with local embedding generation running acceptably on CPU/GPU.

---

## 3. Operational Rebuilding Blueprint

To capitalize on the team's operational work without compromising our core research goals, we establish a strict division of components:

### What We KEEP (The Service Scaffold)
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  FastAPI Structure   │ Async Job Orchestration & persist wrappers for tasks.    │
├──────────────────────┼──────────────────────────────────────────────────────────┤
│  Operational Shell   │ Rate limiting, API key auth, health metrics, and Docker.  │
├──────────────────────┼──────────────────────────────────────────────────────────┤
│  Cypher Validator    │ `assert_readonly_cypher` regex/keyword parser is solid.   │
├──────────────────────┼──────────────────────────────────────────────────────────┤
│  Chunking Signature  │ Keep basic parameters but add sentence-boundary checks.   │
├──────────────────────┼──────────────────────────────────────────────────────────┤
│  Hybrid Fallback     │ Keep the structural fallback pattern: Cypher -> Text.   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### What We REBUILD (Our Core Research Modules)
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Ingestion Pipeline  │ Re-point Hugging Face import job from .en to `20231101.vi`│
├──────────────────────┼──────────────────────────────────────────────────────────┤
│  Entity / Rel Ext.   │ Swap English regex for Vietnamese spaCy/underthesea NER.  │
├──────────────────────┼──────────────────────────────────────────────────────────┤
│  Local Model Layer   │ Re-route embedding & generation prompts to local SLM.    │
├──────────────────────┼──────────────────────────────────────────────────────────┤
│  ReAct Agent Loop    │ Construct dynamic, 6-iteration capped orchestrator loop. │
├──────────────────────┼──────────────────────────────────────────────────────────┤
│  Dataset Generator   │ Build offline template-to-graph extraction pipeline.      │
├──────────────────────┼──────────────────────────────────────────────────────────┤
│  QLoRA Training      │ Build fine-tuning harness for Text-to-Cypher adapters.   │
├──────────────────────┼──────────────────────────────────────────────────────────┤
│  Evaluation Harness  │ Write scripts to calculate joint EM/F1 on ViWiki-MHR.    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Collaborative Team Alignment & Division of Labor

To prevent divergence within the group, we map a clean separation of ownership. This enables team members focused on general service scalability to work in parallel with our sovereign research objectives:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         Core FastAPI Shell & API Routing                      │
│                           (Owned by: Service Team)                            │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│  Operational Engineering     │              │  Research Core & SLM Layer   │
│   - Rate limiting & Docker   │              │   - Vietnamese Tokenization  │
│   - API Key Auth & Metrics   │              │   - Typed Graph Schema       │
│   - Ingestion Job Management  │              │   - Local SLM NF4 Loading    │
│   - Neo4j Server Operations  │              │   - ReAct Agent Logic Loop   │
│                              │              │   - ViWiki-MHR Dataset Gen   │
│  (Owned by: Service Team)    │              │   - QLoRA Adapter Tuning     │
│                              │              │   - Multi-hop Joint Eval     │
│                              │              │                              │
│                              │              │      (Owned by: YOU)         │
└──────────────────────────────┘              └──────────────────────────────┘
```

### Strategic Alignment Ground Rules
1.  **Dual-Mode Model Interface:** Maintain an environment toggle (`MODEL_MODE = "local"` vs `"api"`). This lets the group run quick validation queries against Gemini if they need to test API routes, while keeping your research execution path strictly routed to the local, offline model.
2.  **Dataset Construction Isolation:** Keep the **ViWiki-MHR** dataset construction pipeline as an offline script workstream. Do not run dataset synthesis inside the web api runtime loops. Build the data offline, compile it, and publish it, while using the FastAPI service exclusively for query-time serving.
3.  **Unified Graph Interface:** Extend the Neo4j database client rather than writing a second driver. Agree on the extended nodes and relationships in week 3 so that both teams are querying a unified database, preventing breaking changes from showing up in final system merges.
