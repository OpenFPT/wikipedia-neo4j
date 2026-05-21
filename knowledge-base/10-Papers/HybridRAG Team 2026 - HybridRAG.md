---
title: "Beyond RAG for Cyber Threat Intelligence: A Systematic Evaluation of Graph-Based and Agentic Retrieval"
authors: HybridRAG Authors
year: 2026
url: https://arxiv.org/abs/2604.11419
venue: arXiv preprint
---

tags:: [[paper]], [[hybrid-retrieval]], [[agentic-rag]]

# [[HybridRAG Team 2026 - HybridRAG]]

## TL;DR
A systematic evaluation of four RAG architectures for cyber threat intelligence (CTI) — standard vector retrieval, graph-based retrieval, an agentic variant that repairs failed graph queries, and a hybrid approach combining graph queries with text retrieval — across 3,300 QA pairs. The hybrid graph-text approach improves answer quality by up to 35% on multi-hop questions versus vector-only RAG, while maintaining more reliable performance than graph-only systems.

## Method
The authors build a CTI knowledge graph representing entities (threat actors, malware, vulnerabilities) and their relationships, then implement and evaluate four architectures:
1. **Vector RAG:** Standard dense retrieval over narrative security reports.
2. **Graph RAG:** Structured multi-hop queries over the CTI knowledge graph.
3. **Agentic GraphRAG (AGRAG):** Graph RAG with an automated repair loop for failed queries.
4. **HybridRAG:** Parallel symbolic Cypher queries on the KG combined with keyword + semantic text retrieval, results fused before generation.

Evaluation covers factual lookups, multi-hop relational queries, analyst-style synthesis questions, and unanswerable cases — 3,300 QA pairs total.

## Results
- Graph grounding improves performance on **structured factual queries** over vector-only RAG.
- **HybridRAG improves answer quality by up to 35%** on multi-hop questions compared to vector RAG.
- Graph-only systems are less reliable than hybrid systems due to KG sparsity and failed Cypher queries with no fallback.
- Unanswerable cases remain the hardest category across all architectures.

## Relevance
This directly validates our dual-retrieval architecture and the unanswerable subset design:
- **What we borrow:** The hybrid architecture pattern (parallel Cypher + text retrieval) and the finding that graph-only systems fail on sparse graphs without a text fallback.
- **What we adapt:** Domain — from English CTI to Vietnamese Wikipedia open-domain. The query taxonomy (factual, multi-hop, unanswerable) maps directly onto our ViWiki-MHR split design.
- **What we avoid:** Separate parallel pipeline infrastructure. We implement both retrieval paths as tools inside a single ReAct loop, letting the SLM decide which to call rather than running both unconditionally.
