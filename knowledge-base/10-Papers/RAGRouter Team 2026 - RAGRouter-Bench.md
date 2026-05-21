---
title: "RAGRouter-Bench: A Dataset and Benchmark for Adaptive RAG Routing"
authors: RAGRouter Authors
year: 2026
url: https://arxiv.org/abs/2602.00296
venue: arXiv preprint
---

tags:: [[paper]], [[hybrid-retrieval]], [[query-routing]]

# [[RAGRouter Team 2026 - RAGRouter-Bench]]

## TL;DR
RAGRouter-Bench is the first dataset and benchmark specifically designed for adaptive RAG routing — the problem of selecting which RAG paradigm (e.g. vector, graph, hybrid, agentic) to invoke for a given query-corpus pair. The key finding is that no single RAG paradigm dominates across all query types and corpus structures: adaptive routing yields strictly better effectiveness-efficiency trade-offs than fixed paradigm selection.

## Method
The benchmark is grounded in **query-corpus compatibility**:
- Three canonical query types (factual lookup, multi-hop relational, synthesis/summarization).
- Fine-grained corpus indicators capturing structural and semantic properties of the document collection.
- A unified evaluation protocol measuring both **generation quality** (correctness) and **resource consumption** (latency, cost).

Multiple standard RAG paradigms are implemented with multiple backbone LLMs across all query-corpus combinations. Routing is formulated as context-dependent paradigm selection and evaluated using both quantitative metrics and LLM-as-a-Judge assessments.

## Results
- **No one-size-fits-all paradigm exists** across query-corpus pairs — the best paradigm varies by query type.
- Adaptive routing outperforms any fixed paradigm in both effectiveness and efficiency.
- Query-corpus compatibility is established as a central design principle for next-generation RAG systems.

## Relevance
This paper provides the empirical basis for our intent-based tool routing:
- **What we borrow:** The finding that query type determines optimal retrieval strategy. Single-hop factual queries should route to `text_search`; multi-hop relational queries should route to `kg_query`. This is encoded in our ReAct system prompt as routing heuristics.
- **What we adapt:** The evaluation framework. We apply RAGRouter-Bench's query-type taxonomy (factual, multi-hop, synthesis) to label our ViWiki-MHR test split and measure per-type routing accuracy.
- **What we avoid:** Training a separate neural router model, which would add VRAM and latency overhead. Our orchestrator SLM performs implicit routing via its Thought steps.
