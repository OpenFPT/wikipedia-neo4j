# Applicable Research Papers for Vietnamese Wikipedia GraphRAG

*Generated: 2026-05-26 | Sources: 40+ | Scope: 2023-2026*

## Executive Summary

Key opportunities identified for improving our system:

1. **GraphRAG with Community Detection** — Microsoft's hierarchical Leiden clustering enables both local and global QA
2. **Hybrid Retrieval** — Combining sparse (BM25) + dense + graph traversal dramatically outperforms single-strategy
3. **Vietnamese Embeddings** — Vietnamese-specific models now outperform multilingual alternatives
4. **Inference-Time Scaling** — Sequential + parallel reasoning at query time improves multi-hop accuracy by 64.7%
5. **Better NER** — ViDeBERTa and TextGraphFuseGAT surpass PhoBERT with fewer parameters

---

## 1. GraphRAG & Graph-Augmented Retrieval

### Microsoft GraphRAG (Edge et al., 2024)
- **Paper**: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
- **Key Idea**: Hierarchical community detection (Leiden algorithm) -> community summaries -> map-reduce QA
- **Results**: 70-80% win rate on comprehensiveness/diversity vs. naive RAG
- **Applicability**: Partition our Neo4j graph into topic communities (Vietnamese history, geography, culture). Pre-generate community summaries for thematic queries.
- **Source**: https://arxiv.org/pdf/2404.16130

### Inference-Scaled GraphRAG (Thompson et al., 2025)
- **Paper**: "Inference Scaled GraphRAG: Improving Multi Hop Question Answering on Knowledge Graphs"
- **Key Idea**: Apply inference-time compute scaling (chain-of-thought + majority voting over sampled graph traversals)
- **Results**: 64.7% improvement over traditional GraphRAG; 31.44% vs 15.26% on hard multi-hop
- **Applicability**: Enhance our ReAct agent with parallel trajectory sampling for complex Vietnamese historical questions.
- **Source**: https://arxiv.org/html/2506.19967v1

### HGRAG — Hypergraph RAG (2025)
- **Paper**: "HGRAG: Cross-Granularity Integration for Multi-Hop Question Answering"
- **Key Idea**: Entities as nodes, passages as hyperedges. Hypergraph diffusion propagates entity+passage similarities.
- **Results**: 87.5% F1 on HotpotQA, 6.3x faster retrieval, 59.5% fewer nodes
- **Applicability**: Wikipedia articles naturally map to hyperedges connecting related entities.
- **Source**: https://arxiv.org/pdf/2508.11247

### GRAG — Graph Retrieval-Augmented Generation (2024)
- **Paper**: "GRAG: Graph Retrieval-Augmented Generation"
- **Key Idea**: Divide-and-conquer subgraph retrieval in linear time. Dual-view prompting (text + graph soft prompts).
- **Applicability**: Handles networked documents (Wikipedia hyperlinks). Dual-view prompting helps LLM understand graph structure.
- **Source**: https://arxiv.org/html/2405.16506

### BYOKG-RAG (Mavromatis et al., Amazon, 2025)
- **Paper**: "BYOKG-RAG: Multi-Strategy Graph Retrieval for KGQA"
- **Key Idea**: Combines LLM-based entity linking + path retrieval + OpenCypher queries. Works with custom KGs without training data.
- **Results**: 87.1% Hit on WebQSP, 71.1% on CWQ with only 4.5 LLM calls
- **Applicability**: Multi-strategy approach is ideal for our Neo4j + Cypher system. OpenCypher support is native.
- **Source**: https://aclanthology.org/2025.emnlp-main.1417.pdf

### GraphSearch — Agentic Deep Searching (Yang et al., 2025)
- **Paper**: "GraphSearch: An Agentic Deep Searching Workflow for GraphRAG"
- **Key Idea**: 6-module pipeline (Query Decomposition -> Context Refinement -> Query Grounding -> Logic Drafting -> Evidence Verification -> Query Expansion). Dual-channel retrieval.
- **Applicability**: Directly maps to our ReAct agent architecture. Evidence verification reduces hallucinations.
- **Source**: https://arxiv.org/html/2509.22009

---

## 2. Multi-Hop QA & Graph Reasoning

### GNN-RAG (2024)
- **Paper**: "GNN-RAG: Combining GNNs with RAG for KGQA"
- **Key Idea**: GNN reasons over dense KG subgraph -> extracts shortest paths -> augments LLM retriever
- **Results**: 8.9-15.5% F1 improvement on multi-hop/multi-entity questions
- **Applicability**: GNN subgraph reasoning + path extraction provides interpretable multi-hop chains for our agent.
- **Source**: https://arxiv.org/pdf/2405.20139

### S-Path-RAG (2025)
- **Paper**: "S-Path-RAG: Semantic-Aware Shortest-Path Retrieval for Multi-Hop KGQA"
- **Key Idea**: Bounded-length candidate paths (hybrid k-shortest + beam + constrained random-walk). Differentiable path scorer.
- **Results**: 78.2% F1 on CWQ, 3.8ms latency, stable for 5-8 hop questions
- **Applicability**: Semantic path weighting prioritizes culturally relevant connections in Vietnamese Wikipedia.
- **Source**: https://arxiv.org/html/2603.23512

### A*Net — Scalable Path-Based Reasoning (NeurIPS 2023)
- **Paper**: "A*Net: Scalable Path-Based KG Reasoning"
- **Key Idea**: A* algorithm searches important paths only. SOTA on million-scale KGs using 0.2% of nodes/edges.
- **Applicability**: Critical for scaling to full Vietnamese Wikipedia (1.6M articles).
- **Source**: https://papers.neurips.cc/paper_files/paper/2023/file/b9e98316cb72fee82cc1160da5810abc-Paper-Conference.pdf

### Think-on-Graph 2.0 (ToG-2) (2024)
- **Paper**: "Think-on-Graph 2.0: Tight-Coupling Hybrid RAG"
- **Key Idea**: Iteratively couples KG-based and text-based retrieval. Training-free.
- **Applicability**: Alternates between graph traversal and dense retrieval — maps to our Cypher + fulltext fallback.
- **Source**: http://arxiv.org/pdf/2407.10805v5

### CoRAG — Cooperative Retriever (EMNLP 2025)
- **Paper**: "CoRAG: Cooperative Hybrid Retriever"
- **Key Idea**: Dynamically chooses between textual search and graph traversal. Global retrieval bypasses locality constraints.
- **Applicability**: Solves locality problem in graph-only retrieval.
- **Source**: https://aclanthology.org/2025.findings-emnlp.872.pdf

---

## 3. Hybrid Retrieval Strategies

### Blended RAG (2024)
- **Paper**: "Blended RAG: Semantic Search + Hybrid Queries"
- **Key Idea**: BM25 (sparse) + dense KNN + Elastic Learned Sparse Encoder. Best Fields strategy.
- **Results**: 87% retriever accuracy on TREC-COVID
- **Source**: https://arxiv.org/pdf/2404.07220v2

### WeKnow-RAG (2024)
- **Paper**: "WeKnow-RAG: Web Search + KG Integration"
- **Key Idea**: Multi-stage: sparse (BM25) -> hybrid (BM25 + dense) -> reranking
- **Applicability**: Two-stage pattern with cross-encoder reranking (we already have BAAI/bge-reranker-v2-m3).
- **Source**: https://arxiv.org/pdf/2408.07611

### Mix-of-Granularity (MoG/MoGG) (2024)
- **Paper**: "Mix-of-Granularity: Dynamic Chunking for RAG"
- **Key Idea**: Router dynamically selects optimal chunk granularity per query.
- **Applicability**: Optimizes chunk size for Vietnamese Wikipedia articles of varying lengths.
- **Source**: https://arxiv.org/html/2406.00456v2

### Mixture-of-PageRanks (2024)
- **Paper**: "Mixture-of-PageRanks: Sparse GraphRAG"
- **Key Idea**: TF-IDF embeddings + personalized PageRank. CPU-only, memory-efficient.
- **Applicability**: Cost-effective for large-scale Wikipedia. Complements dense retrieval.
- **Source**: https://arxiv.org/html/2412.06078v1

---

## 4. Vietnamese NLP Advances

### ViDeBERTa (2023)
- **Paper**: "ViDeBERTa: Pre-trained Language Model for Vietnamese"
- **Results**: 95.3% F1 on PhoNER (vs. 94.7% PhoBERT-large), 23% fewer parameters
- **Applicability**: Drop-in replacement for PhoBERT in NER backend.
- **Source**: https://ar5iv.labs.arxiv.org/html/2301.10439

### TextGraphFuseGAT (2024)
- **Paper**: "TextGraphFuseGAT: Graph-Enhanced NER for Vietnamese"
- **Results**: 98.4% Micro-F1 on PhoNER-COVID19 (vs. 94.5% PhoBERT-large)
- **Applicability**: Graph-enhanced NER architecture shows huge gains with domain-specific fine-tuning.
- **Source**: https://arxiv.org/pdf/2510.11537

### Vietnamese Embedding Models (2024-2025)

| Model | Dims | Metric | Score | Notes |
|-------|------|--------|-------|-------|
| Vietnamese_Embedding_v2 (AITeamVN) | 1024 | Acc@10 | 95.78% | Fine-tuned BGE-M3, 1.1M triplets |
| GreenNode-Embedding-Large-VN-Mixed | 1024 | MAP@5 | 42.08 | XLM-R-based, 8192 token context |
| dangvantuan/vietnamese-embedding | 768 | Pearson | 88.33 | PhoBERT-based, SOTA STSB-vn |
| Halong Embedding | 768 | MAP@5 | 32.15 | multilingual-e5-base fine-tuned |
| Vietnamese-bi-encoder (BKAI) | 768 | Acc@1 | 73.28% | PhoBERT-base, MS MARCO trained |

**Recommendation**: Use `Vietnamese_Embedding_v2` or `GreenNode-Embedding-Large-VN-Mixed` for RAG. GreenNode supports 8K token context.

### UIT-ViQuAD 2.0 (Nguyen et al., 2022)
- **Dataset**: 35,990 QA pairs (23K answerable + 12K adversarial unanswerable)
- **Best**: 77.24% F1 (XLM-RoBERTa), Human: ~87%
- **Source**: https://ar5iv.labs.arxiv.org/html/2203.11400

---

## 5. Knowledge Graph Completion & Link Prediction

### KICGPT (EMNLP 2023)
- **Key Idea**: In-context learning with "Knowledge Prompts" encoding structural KG info. No fine-tuning.
- **Applicability**: Backfill missing Wikipedia entity relationships using LLM prompting.
- **Source**: https://aclanthology.org/2023.findings-emnlp.580.pdf

### Link Prediction as NLI (UIT-NLP, DSAA 2023)
- **Key Idea**: Wikipedia link prediction as sentence pair classification.
- **Results**: 0.99996 Macro F1 on DSAA-2023 competition
- **Applicability**: Vietnamese team's approach — directly applicable for improving hyperlink coverage.
- **Source**: https://ar5iv.labs.arxiv.org/html/2308.16469

### Zero-Shot Link Prediction (CTLP, 2024)
- **Key Idea**: Condensed Transition Graph, encodes all-paths in linear time. Contrastive learning.
- **Applicability**: Predicts missing Wikipedia links without training data.
- **Source**: https://arxiv.org/html/2402.10779v2

---

## 6. Evaluation Metrics for GraphRAG

### From Microsoft GraphRAG:
- **Comprehensiveness**: Does the answer cover all aspects?
- **Diversity**: Does it provide varied perspectives?
- **Empowerment**: Does it help the user understand?
- **Claim-based evaluation**: Decompose answers into atomic claims, verify each

### From RAGAs Framework:
- **Context Relevance**: Are retrieved chunks relevant?
- **Faithfulness**: Is the answer grounded in context?
- **Answer Correctness**: Does it match ground truth?

### Recommended Stack:
1. Retrieval: Context Hit Rate, MRR, NDCG@k (already implemented)
2. Generation: Faithfulness (claim decomposition), Answer F1
3. End-to-end: Multi-hop accuracy on VIMQA/ViQuAD 2.0
4. Latency: Per-hop retrieval time, total response time

---

## 7. Implementation Priority Matrix

| Priority | Technique | Effort | Impact | Paper |
|----------|-----------|--------|--------|-------|
| **P0** | Vietnamese-specific embeddings | Low | High | Vietnamese_Embedding_v2 |
| **P0** | Hybrid retrieval (BM25 + dense + graph) | Medium | High | Blended RAG, ToG-2 |
| **P1** | Leiden community detection | Medium | High | Microsoft GraphRAG |
| **P1** | Multi-strategy retrieval (entity + path + Cypher) | Medium | High | BYOKG-RAG |
| **P1** | ViDeBERTa NER backend | Low | Medium | ViDeBERTa |
| **P2** | Inference-time scaling for agent | Medium | Medium | Inference-Scaled GraphRAG |
| **P2** | Hypergraph structure | High | Medium | HGRAG |
| **P2** | Link prediction backfill | Medium | Medium | KICGPT, CTLP |
| **P3** | Dynamic chunk granularity | Medium | Low | MoG |
| **P3** | Graph-enhanced NER | High | Medium | TextGraphFuseGAT |

---

## 8. Recommended Reading Order

1. **Microsoft GraphRAG** — foundational architecture
2. **BYOKG-RAG** — practical multi-strategy retrieval for Neo4j
3. **Think-on-Graph 2.0** — tight coupling of graph + text retrieval
4. **Blended RAG** — hybrid retrieval implementation details
5. **Inference-Scaled GraphRAG** — scaling reasoning at query time
6. **Vietnamese_Embedding_v2 / GreenNode** — embedding model selection

---

## Full Source List

1. [GraphRAG: From Local to Global](https://arxiv.org/pdf/2404.16130) — Microsoft, 2024
2. [Inference Scaled GraphRAG](https://arxiv.org/html/2506.19967v1) — NCSU, 2025
3. [HGRAG: Hypergraph RAG](https://arxiv.org/pdf/2508.11247) — 2025
4. [GRAG](https://arxiv.org/html/2405.16506) — 2024
5. [BYOKG-RAG](https://aclanthology.org/2025.emnlp-main.1417.pdf) — Amazon, EMNLP 2025
6. [GraphSearch](https://arxiv.org/html/2509.22009) — 2025
7. [GNN-RAG](https://arxiv.org/pdf/2405.20139) — 2024
8. [S-Path-RAG](https://arxiv.org/html/2603.23512) — 2025
9. [A*Net](https://papers.neurips.cc/paper_files/paper/2023/file/b9e98316cb72fee82cc1160da5810abc-Paper-Conference.pdf) — NeurIPS 2023
10. [Think-on-Graph 2.0](http://arxiv.org/pdf/2407.10805v5) — 2024
11. [CoRAG](https://aclanthology.org/2025.findings-emnlp.872.pdf) — EMNLP 2025
12. [Blended RAG](https://arxiv.org/pdf/2404.07220v2) — 2024
13. [WeKnow-RAG](https://arxiv.org/pdf/2408.07611) — 2024
14. [Mix-of-Granularity](https://arxiv.org/html/2406.00456v2) — 2024
15. [Mixture-of-PageRanks](https://arxiv.org/html/2412.06078v1) — 2024
16. [ViDeBERTa](https://ar5iv.labs.arxiv.org/html/2301.10439) — 2023
17. [TextGraphFuseGAT](https://arxiv.org/pdf/2510.11537) — 2024
18. [UIT-ViQuAD 2.0](https://ar5iv.labs.arxiv.org/html/2203.11400) — 2022
19. [KICGPT](https://aclanthology.org/2023.findings-emnlp.580.pdf) — EMNLP 2023
20. [Link Prediction as NLI](https://ar5iv.labs.arxiv.org/html/2308.16469) — DSAA 2023
21. [Zero-Shot Link Prediction](https://arxiv.org/html/2402.10779v2) — 2024
22. [PhoNLP](https://aclanthology.org/2021.naacl-demos.1.pdf) — NAACL 2021
23. [Vietnamese-bi-encoder](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder) — 2024
24. [Towards Comprehensive Vietnamese RAG](https://arxiv.org/abs/2403.01616) — 2024
25. [Power-Link](https://arxiv.org/html/2401.02290) — 2024
26. [SR-GNN for KGC](https://link.springer.com/article/10.1007/s10489-024-05482-2) — 2024
27. [MPIKGC](https://aclanthology.org/2024.lrec-main.1044.pdf) — LREC 2024
