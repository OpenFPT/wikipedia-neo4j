---
title: "HiGraAgent: Dual-Agent Adaptive Reasoning over Hierarchical Knowledge Graph for Open Domain Multi-hop Question Answering"
authors: Hung Luu, Long S. T. Nguyen, Trung Pham, Hieu Pham, Tho Quan
year: 2026
url: https://aclanthology.org/2026.findings-eacl.62
venue: Findings of EACL 2026
doi: 10.18653/v1/2026.findings-eacl.62
---

tags:: [[paper]], [[agentic-rag]], [[knowledge-graph]]

# [[Luu et al. 2026 - HiGraAgent]]

## TL;DR
HiGraAgent addresses the dual compositionality challenge in open-domain multi-hop QA — reasoning over complex query structures while integrating scattered evidence. It constructs a bi-layer hierarchical KG (Entity Layer + Passage Layer), uses a hybrid graph-semantic retriever, and orchestrates two agents (Seeker and Librarian) in an iterative protocol. Achieves 85.3% average accuracy across HotpotQA, 2WikiMultihopQA, and MuSiQue — surpassing the strongest prior system by 11.7%.

## Method
Three components:
1. **HiGra (Hierarchical Knowledge Graph):** A bi-layer graph — an Entity Layer capturing typed entity relations, and a Passage Layer linking entity nodes to their source text passages. An entity alignment step reduces KG redundancy by **34.5%** while preserving expressiveness.
2. **HiGraRetriever:** A hybrid retriever that combines graph traversal (structural) and semantic embedding search (dense) in a single retrieval step, outperforming traditional graph-only methods on single-step reasoning.
3. **Dual-Agent Adaptive Reasoning (Seeker–Librarian):** A Seeker agent plans graph traversal paths and formulates sub-queries; a Librarian agent retrieves and validates passage-level evidence for each sub-query. They iterate until a sufficient answer is assembled.

## Results
- **85.3% average accuracy** across HotpotQA, 2WikiMultihopQA, MuSiQue.
- Surpasses strongest prior system by **11.7 pp**.
- Entity alignment reduces KG redundancy by **34.5%** without loss of expressiveness.
- HiGraRetriever outperforms traditional graph-only retrieval on single-step reasoning sub-tasks.

## Relevance
Validates our hierarchical schema design and informs our tool architecture:
- **What we borrow:** The bi-layer graph design principle — structured entity-relation edges (Entity Layer) directly linked to raw text chunk nodes (Passage Layer). This is exactly our Neo4j schema: `(:Entity)-[:MENTIONED_IN]->(:Chunk)-[:PART_OF]->(:Page)`.
- **What we adapt:** The dual-agent structure. A two-agent system is computationally too heavy for a local SLM. We collapse Seeker and Librarian roles into a single orchestrator alternating between `kg_query` (Seeker behavior) and `get_passage` (Librarian behavior) within one ReAct loop.
- **What we avoid:** The entity alignment preprocessing step at query time — we perform alignment offline during KG construction.
