# Vietnamese KGQA System Design: ViWiki-MHR Pipeline

This document establishes the architecture, data flow, tool specifications, dataset parameters, fine-tuning recipes, and execution roadmap for our fully-local, single-process Vietnamese Knowledge Graph Question Answering (KG-QA) and Retrieval-Augmented Generation (RAG) system.

---

## 1. System Architecture

The pipeline is designed as a **fully-local, single-process, privacy-preserving system** that runs entirely on consumer-grade developer hardware without outbound network dependencies.

### Target Hardware Specification
*   **System RAM:** 16GB
*   **VRAM:** 8GB (RTX 3060/4060 class) or Apple Silicon with 16GB Unified Memory.
*   **Quantization:** Base Small Language Model (SLM) runs in 4-bit NormalFloat (NF4) to stay within the 8GB VRAM cap.
*   **Embeddings:** Small multilingual models (e.g., `bge-m3`) running acceptably on CPU.

```
+-----------------------------------------------------------------------------------+
|                              Local Consumer Device                                |
|                                                                                   |
|  +--------------------+      +-------------------------------------------------+  |
|  |    User Prompt     | ---> |               SLM Orchestrator                  |  |
|  | (Vietnamese Query) |      | (Qwen 2.5-7B / SeaLLM-7B in 4-bit NF4 VRAM)     |  |
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
|           |   Server      |   |   + BM25      |   | Paragraph     |               |
|           |  (Graph KG)   |   | (Vector CPU)  |   | Local Dump    |               |
|           +---------------+   +---------------+   +---------------+               |
+-----------------------------------------------------------------------------------+
```

---

## 2. Components & Storage Layer

*   **Frontend:** A minimal, command-line interface (CLI) for development, with a Gradio-based web interface added during polishing phases. It displays the raw user query, the reasoning/tool execution trace, the cited Wikipedia paragraphs, and the final answer.
*   **Orchestrator (Fine-Tuned SLM):** A 7B-parameter instruction-tuned model running the agent loop. It parses system prompts, makes tool decisions, processes observations, and synthesizes cited answers.
*   **Knowledge Graph (KG) Store:** A local **Neo4j** database server running on the same machine. It stores entities, relations, and timestamps extracted from a pinned Vietnamese Wikipedia snapshot. Neo4j provides native Cypher query support, built-in full-text indexing, and vector search — making it the natural target for our Text-to-Cypher fine-tuning pipeline.
*   **Vector Store:** **Qdrant** (local mode) or a flat **FAISS** index storing dense embeddings of Wikipedia paragraphs for semantic fallback.
*   **BM25 Index:** A sparse lexical search index compiled over the same Wikipedia paragraph corpus to handle entity-heavy Vietnamese queries, serving as a low-cost, high-precision retrieval fallback.

---

## 3. Tool Specifications

To prevent Small Language Models from experiencing decision confusion, we strictly cap the agent's actions to four clear, deterministic tools:

### `kg_schema()`
*   **Input:** None.
*   **Output:** JSON schema outlining active node labels, relationship types, and property names.
*   **Optimization:** Results are cached after the first execution to bypass database overhead.

### `kg_query(cypher: str)`
*   **Input:** A schema-compliant Cypher query string.
*   **Output:** Result rows returned by the graph database. If compilation or matching fails, raw database error strings are piped back as an observation so the agent can self-correct.

### `text_search(query: str, k: int)`
*   **Input:** Search term (`query`) and number of chunks (`k`).
*   **Output:** Hybrid reranked retrieve (BM25 lexical + dense vector CPU) returning the top-k passages with their respective unique `passage_ids`.

### `get_passage(passage_id: str)`
*   **Input:** Unique passage identifier.
*   **Output:** Full raw text of the Wikipedia paragraph, used by the orchestrator to verify facts or resolve entity context gaps prior to final answer synthesis.

---

## 4. ReAct Loop & Citation Tracking

```
User Query
   │
   ▼
[System Prompt: Tools, Schema, Instructions]
   │
   ▼
Thought: "I need to find X in the database..."
   │
   ▼
Action: {"tool": "kg_query", "args": {"cypher": "..."}} (JSON format)
   │
   ▼
Observation: [Tool Output / Database Compile Error]
   │
   ▼
(Loop Capped at 6 Max Iterations) ───[If Iteration >= 6]───► Force Answer Synthesis
   │                                                               │
   ▼ [If Answer Emitted]                                           ▼
Final Answer: "..." containing [passage_id] citations ◄────────────┘
   │
   ▼
[Groundedness Post-Processor Verification]
   ├──► Verify all cited [passage_id] tags exist in retrieval history.
   ├──► Pass: Output response with cited text snippets.
   └──► Fail: Flag answer as UNGROUNDED.
```

---

## 5. The ViWiki-MHR Dataset Spec

To maintain an open-science posture, **ViWiki-MHR** (30K+ examples) is fully independently constructed, bypassing gated datasets like ViMQA entirely. Access to ViMQA is restricted by a user agreement requiring bilateral correspondence, which limits downstream commercial reuse and reproducible research. We position ViWiki-MHR as an **open alternative** released under a CC-BY-SA license on Hugging Face.

### Dataset Composition
We combine open canonical data from UIT-ViQuAD 2.0 (verified license) with newly synthesized, KG-grounded multi-hop reasoning and unanswerable subsets:

| Dataset Component | Source | Hops | Answerable | Target Size | Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Single-hop Answerable** | `uit-viquad` | 1 | Yes | ~23,000 | Reformatted from UIT-ViQuAD 2.0 |
| **Single-hop Unanswerable** | `uit-viquad` | 1 | No | ~5,000 | Subsampled from UIT-ViQuAD 2.0 adversarial set |
| **Multi-hop Answerable** | `synthetic-multihop` | 2 or 3+ | Yes | ~7,000 | Generated via KG walks & frontier LLM rewriting |
| **Adversarial Multi-hop Unanswerable** | `synthetic-multihop` | 2 or 3+ | No | ~1,000 | Broken-link walks (first Vietnamese benchmark) |

### The Adversarial Multi-hop Unanswerable Contribution
While UIT-ViQuAD 2.0 provides high-quality, crowdwritten single-hop adversarial unanswerables, **multi-hop unanswerability is a major benchmark gap** in Vietnamese. We introduce the first benchmark subset targeting this failure mode:
*   **Construction:** We walk a valid n-hop chain in our local KG, and intentionally **break one link** (replacing a target entity, swapping a relation type, or introducing a factual contradiction).
*   **Characteristics:** The surface question looks completely coherent and valid up to hop 2, but fails at hop 3. This forces the agentic loop to verify each path step programmatically, explicitly outputting *"I don't know"* if a link fails, preventing confidently hallucinated fabrications.

```
Valid Path Walk:   (Ngô Tất Tố) ─[:SÁNG_TÁC]─► (Tắt đèn) ─[:BỐ_CẢNH]─► (Cẩm Giàng)
Broken Link Walk:  (Ngô Tất Tố) ─[:SÁNG_TÁC]─► (Tắt đèn) ─[:BỐ_CẢNH]─► [Fabricated: Hải Phòng]
                                                                        (KG & Text check fails)
```

---

## 6. Model Fine-Tuning Pipeline

We perform parameter-efficient fine-tuning via **QLoRA** over our selected base instruct-tuned SLM (e.g. `Qwen/Qwen2.5-7B-Instruct` or `vinai/PhoGPT-7B`).

### Task 1: Vietnamese Text-to-Cypher (Primary Target)
*   **Purpose:** Fine-tunes the SLM to write schema-compliant Cypher queries from Vietnamese questions.
*   **Data Size:** ~10-15K generated pairs.
*   **Quality Control:** Every generated Cypher query must compile and return non-empty entities against our local KG store during processing; failed samples are filtered out.
*   **Evaluation:** 500 held-out, human-verified pairs drawn from the test split.

### Task 2: Agent Action & Synthesis (Secondary Target)
*   **Purpose:** Fine-tunes the model's ReAct format conformance, forcing it to generate structured JSON blocks (`Action: {"tool": "...", "args": {...}}`), follow the 6-iteration cap, and synthesize answers grounded in cited observations.

### Training Configuration
*   **Bits:** 4-bit NormalFloat (NF4).
*   **Double Quantization:** Enabled.
*   **Target Modules:** All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
*   **LoRA Rank ($r$):** 32
*   **LoRA Alpha ($\alpha$):** 64
*   **Optimizer:** Paged AdamW.

---

## 7. Execution Plan & Phase Milestones

```
- Foundations, Bake-off, 4-bit SLM running locally, Obsidian setup
Entity/Relation Extraction, Neo4j database loading, 4 Tools built
- Programmatic template question gen, LLM rewriting, first 5K QC proof
- Base SLM agent running on 100 questions (working pipeline check)
- Full Text-to-Cypher QLoRA fine-tuning, adapter comparison tests
- Full 30K dataset generation, Hugging Face prepare, test set human verification
- System Evaluation: run agentic system vs baselines on full test set, generate tables
- Paper writing: frame local deployment / sovereignty argument, compile results
- Code GitHub publish (one-line install), Hugging Face datasets & model release, SUBMIT!
```

### Phase Milestones
*   **Phase 1 Milestone: Base SLM** - Base SLM candidate selected, running in 4-bit locally, and first hello-world text generated.
*   **Phase 2 Milestone: End-to-End Baseline** - Local KG store populated and end-to-end ReAct pipeline running on a 100-question sample.
*   **Phase 3 Milestone: Text-to-Cypher Adapter** - Fine-tuned QLoRA Text-to-Cypher adapter outperforming the base SLM by a measurable margin.
*   **Phase 4 Milestone: Comprehensive Evaluation** - Benchmark evaluation metrics compiled, results tables completed, and system compared against baseline modes.

### Strategic Mentor Verification Checklist
1.  **Gated baseline positioning:** Confirm the capability comparison stance. We cite ViMQA as a restricted access baseline requiring bilateral permission, positioning ViWiki-MHR as an open complementary alternative. No ViMQA data is used.
2.  **UIT-ViQuAD 2.0 Integration:** Confirm canonical license terms for the UIT-NLP corpus to ensure compliant derivative re-release of our 1-hop subsets.
3.  **Adversarial multi-hop unanswerable subset:** Confirm the 1K-2K target for the broken-link KG walks to validate hallucination checks.
4.  **KG Store Select:** Neo4j chosen for native Cypher support, mature tooling, and direct compatibility with CyVerACT's validation pipeline.
5.  **Exclusion of Web Search:** Verified that web retrieval is excluded from scope to preserve fully-local privacy.
