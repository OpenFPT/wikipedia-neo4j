---
title: "CyVerACT: An Open-Source, LLM-Agnostic Evaluator-Optimizer Workflow for Deterministic Text-to-Cypher Verification"
authors: Christina-Maria Androna, Ioanna Mandilara, Eleftheria Arkadopoulou, et al.
year: 2026
url: https://doi.org/10.1016/j.ipm.2026.104836
venue: Information Processing & Management (Vol. 63, Issue 7, 2026)
license: CC BY-SA 4.0
repo: https://gitlab.com/netmode/CyVerACT
---

tags:: [[paper]], [[agentic-rag]], [[multilingual-text2cypher]], [[knowledge-graph]]

# [[Androna et al. 2026 - CyVerACT]]

## TL;DR
CyVerACT is an open-source, LLM-agnostic agentic workflow implemented in LangGraph that leverages the CyVer library to deterministically validate and refine generated Cypher queries against a target Knowledge Graph schema. By replacing standard stochastic LLM-based evaluators with a deterministic syntax, schema, and property validation engine, CyVerACT guides query optimization using actionable compiler-style error metadata and dynamic schema fallbacks.

## Method
The authors construct a structured, directed evaluator-optimizer graph using LangGraph:
1.  **KG Schema Filtering:** Narrows down the target KG schema to relevant entities based on the natural language user query.
2.  **Cypher Generator:** Prompts the LLM with the user query and the filtered schema to generate a candidate Cypher query.
3.  **CyVer Evaluator:** A deterministic regular-expression tool containing three validators:
    *   *Syntax Validator:* Validates raw Cypher syntax correctness.
    *   *Schema Validator:* Ensures generated paths comply with valid node labels and directed relationship types.
    *   *Properties Validator:* Verifies property keys against node/edge schema definitions.
4.  **Cypher Corrector:** Propagates detailed error metadata, error codes, and actionable repair recommendations (e.g. *Wrong Direction Path* warning showing the corrected arrows) back to the LLM for iterative correction.
5.  **Fallback Schema Retriever:** If the model fails to produce an error-free query within a configured retry limit, the system falls back to retrieving the *complete* schema and performing a final generation pass.

## Results
*   **Benchmarks:** Evaluated on the Neo4j Benchmark and the highly complex SustainGraph dataset.
*   **Target Models:** Benchmarked across `GPT-4o`, `Qwen2.5-72B-Instruct`, `Qwen2.5-Coder-32B-Instruct`, `Gemma2-9B-it`, and `Llama-3.1-8B-Instruct` (including the fine-tuned Azzede variant).
*   **Accuracy Improvements:**
    *   Boosted **KG Valid** query execution rates by up to **52.7%** (for fine-tuned Azzede LLaMA-3.1-8B) and enhanced **Positive Jaccard** output rates by up to **31.6%** compared to standard Single-shot setups.
    *   Improved `GPT-4o`'s **KG Valid** query rate by **13.5%** on the SustainGraph dataset.
    *   Open-source models like `Qwen2.5-Coder-32B-Instruct` and `Qwen2.5-72B-Instruct` matched or exceeded `GPT-4o`'s performance on SustainGraph, proving that schema-constrained, deterministic validation bridges the performance gap between smaller edge models and proprietary commercial APIs.
*   **Regressions Avoided:** Demonstrated that standard LLM-based evaluators (such as LangChain's basic Cypher agents) introduce severe performance regressions (false positives/negatives and JSON formatting errors), validating the need for deterministic, schema-constrained checkers.

## Relevance
This paper provides the exact architectural blueprint for our query generation and error-mitigation workflow:
*   **What we borrow:** The concept of execution-aware, schema-constrained error feedback. We implement CyVer's deterministic checking pattern directly inside our orchestrator tools.
*   **What we adapt:** The LangGraph nodes. While CyVerACT is written as a multi-node LangGraph structure, we collapse this into a unified **ReAct orchestrator loop** where our local SLM (`Sailor2-8B`) calls our database query tool (`kg_query`), receives the deterministic syntax/schema error outputs, and corrects the Cypher query dynamically within its context window.
*   **What we avoid:** Heavy enterprise Milvus/LangGraph cluster overheads, executing our check rules locally and synchronously on our Neo4j engine.
