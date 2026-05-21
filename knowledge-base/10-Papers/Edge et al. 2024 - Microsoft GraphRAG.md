---
title: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
authors: Darren Edge et al. (Microsoft Research)
year: 2024
url: https://arxiv.org/pdf/2404.16130
venue: arXiv preprint
---

tags:: [[paper]], [[graph-rag]], [[knowledge-graph]], [[summarization]]

# [[Edge et al. 2024 - Microsoft GraphRAG]]

## TL;DR
Microsoft's GraphRAG is a retrieval-augmented generation framework designed to support query-focused summarization over large, unstructured text corpora. While traditional RAG struggles with high-level, global synthesis questions (e.g., thematic summaries), GraphRAG overcomes this by offline structuring of the corpus into an entity-relation knowledge graph and generating hierarchical summaries of its community clusters.

## Method
The authors built a two-stage pipeline. In the offline indexing phase, an LLM processes source documents to extract entity-relation triples (nodes, edges, and semantic descriptions), builds a unified knowledge graph, clusters nodes into a multi-level hierarchy using the Leiden community detection algorithm, and pre-generates summaries for each community. At query runtime, the system uses two separate search engines: Global Search (a Map-Reduce flow aggregating community summaries at a specified level) and Local Search (extracting specific entity neighborhoods and associated raw source chunks).

## Results
*   **Token Scalability:** Effectively summarized datasets containing upwards of 1 million tokens.
*   **Quality Metrics:** Outperformed standard naive RAG on multi-perspective comprehensiveness, diversity of generated ideas, and factual accuracy.
*   **Empirical Ratios:** Leiden community partitioning consistently preserved modularity across varying community sizes, reducing final generation context size requirements compared to feeding raw source text.

## Relevance
This work establishes the dual search paradigm (Local vs. Global Search).
*   **What we borrow:** The entity-relation knowledge graph and the local search paradigm. In our architecture, the agent triggers local sub-graph traversals using the `kg_query(cypher)` tool.
*   **What we adapt:** Hardware targeting. While Microsoft's GraphRAG requires substantial enterprise GPU setups, we design our entire indexing and inference loop to run on a single **fully-local consumer device** (e.g. 16GB system RAM, 8GB VRAM like an RTX 3060/4060 or Apple Silicon).
*   **What we avoid:** Leiden-based Global Search. Parallel Map-Reduce summarization requires massive API token overhead and context windows, which easily crash or clog 7B-parameter models in local 4-bit quantization. We rely on deterministic Cypher lookups and precise hybrid text fallback to bypass this cost entirely.

```mermaid
graph TD
    subgraph Indexing Pipeline (Offline)
        Docs[Source Documents] --> Chunking[Text Chunking]
        Chunking --> TripleExtraction[LLM Entity & Relation Extraction]
        TripleExtraction --> GraphBuild[Construct Raw Knowledge Graph]
        GraphBuild --> CommunityDetect[Hierarchical Leiden Clustering]
        CommunityDetect --> ClusterSummaries[LLM-generated Community Summaries]
    end
    
    subgraph Querying Pipeline (Online)
        UserQuery([User Query]) --> SearchType{Query Scope?}
        SearchType -->|Global Search| GlobalEngine[Map-Reduce Community Summaries]
        SearchType -->|Local Search| LocalEngine[Retrieve Entity-Relation Subgraph]
        GlobalEngine --> FinalSynthesize1[Final Response]
        LocalEngine --> FinalSynthesize2[Final Response]
    end
```
