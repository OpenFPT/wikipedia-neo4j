---
title: "Mitigating KG Quality Issues: A Robust Multi-Hop GraphRAG Retrieval Framework"
authors: C2RAG Authors
year: 2026
url: https://arxiv.org/abs/2603.14828
venue: arXiv preprint
---

tags:: [[paper]], [[hybrid-retrieval]], [[knowledge-graph]]

# [[C2RAG Team 2026 - C2RAG]]

## TL;DR
C2RAG (Constraint-Checked Retrieval-Augmented Generation) addresses two failure modes of GraphRAG over imperfect KGs: retrieval drift caused by spurious noise, and retrieval hallucinations caused by incomplete information. It proposes constraint-based structural retrieval to filter spurious candidates, and a sufficiency check that triggers a textual recovery fallback when retrieved graph evidence is insufficient to propagate to the next hop.

## Method
Two components operate in sequence:
1. **Constraint-Based Retrieval:** Decomposes each multi-hop query into atomic constraint triples. Uses fine-grained constraint anchoring — matching each constraint against candidate paths — to filter out spurious noisy edges that would cause retrieval drift.
2. **Sufficiency Check:** Before propagating retrieved evidence to the next reasoning hop, evaluates whether the current sub-graph contains enough structural information. If not sufficient, activates **textual recovery** — falling back to unstructured text retrieval to fill the evidence gap — rather than hallucinating a missing link.

## Results
- Consistently outperforms latest GraphRAG baselines by **+3.4% EM** and **+3.9% F1** on average across multi-hop benchmarks.
- Exhibits improved robustness specifically under KG quality issues (noise, missing triples) compared to methods that assume a clean graph.

## Relevance
The sufficiency-check mechanism directly maps onto our fallback routing logic:
- **What we borrow:** The principle of explicit sufficiency gating — before the orchestrator writes a final answer, it must confirm the graph evidence is non-empty and logically connected. If `kg_query` returns zero paths, route to `text_search` rather than generating a fabricated answer.
- **What we adapt:** Implementation level. C2RAG implements this as a dedicated pipeline module. We implement the same logic as a conditional branch inside the ReAct loop prompt and within the `verify_tool` check.
- **What we avoid:** The constraint-triple decomposition preprocessing pipeline, which requires a structured query planner. We rely on our fine-tuned SLM to implicitly decompose queries via its Cypher generation.
