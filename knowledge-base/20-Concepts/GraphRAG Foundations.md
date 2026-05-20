# GraphRAG Foundations

GraphRAG combines the semantic capabilities of traditional vector search with the structured relationships of Knowledge Graphs (KGs). It provides a dual-engine architecture designed to handle both highly specific facts (Local Search) and high-level syntheses (Global Search).

---

## Naive RAG vs. GraphRAG

Traditional "naive" RAG treats documents as a flat list of text chunks. While effective for simple fact extraction, it fails on complex queries.

| Feature | Naive RAG | GraphRAG (Local Search) | GraphRAG (Global Search) |
| :--- | :--- | :--- | :--- |
| **Data representation** | Flat text chunks (vector index) | Graph Nodes, Edges, Properties + Text chunks | Hierarchical Community Summaries (Leiden Clusters) |
| **Query focus** | Single-point semantic similarity | Multi-hop Entity Neighbors & Relations | Dataset-wide themes, summaries, & viewpoints |
| **Multi-hop reasoning** | Poor (relies on overlapping chunks) | **Excellent** (traverses explicit graph edges) | Moderate (aggregates broad community concepts) |
| **Context windows** | Large (feeds raw text chunks) | Small (feeds structured sub-graphs + key facts) | Medium (feeds pre-generated summaries) |
| **Computational cost** | Low (single embedding lookup) | Medium (entity extraction + graph queries) | High (offline Leiden clustering + LLM summaries) |

---

## The Dual Search Engines

Microsoft's GraphRAG framework defines two core methods for query execution:

### 1. Local Search (Entity-Relation Neighborhoods)
*   **How it works:** Extracted entities from the query are matched against the Graph Database. The engine retrieves the target entity, its direct structural neighbors, connected relationships (predicates), and the original text chunks where those connections were found.
*   **Why it shines:** It provides absolute factual grounding. By combining structured graph paths (e.g., `(A)-[:B]->(C)`) with unstructured source text, the LLM has all the exact context required to answer, completely eliminating hallucinated associations.
*   **Relevance to ViWiki-MHR:** This is the primary engine we will use for evaluations. Multi-hop questions are structurally identical to traversing multi-step relationships.

### 2. Global Search (Hierarchical Leiden Summaries)
*   **How it works:** The engine divides the entire Knowledge Graph into communities using the Leiden algorithm (clustering based on modularity). An LLM pre-generates a summary for every community at multiple levels of granularity. When a query is made, a Map-Reduce flow aggregates these summaries.
*   **Why it shines:** It is designed for Query-Focused Summarization (QFS). It can synthesize data-wide perspectives without reading millions of raw tokens at runtime.

---

## Optimization for Consumer Hardware

While Microsoft's enterprise setup utilizes heavy multi-step Leiden clustering and expensive online global Map-Reduce loops, our fully-local system adapts these paradigms to fit a consumer hardware budget (16GB system RAM, 8GB VRAM RTX 3060/4060):

1.  **Direct Cypher Traversal:** Local search is mapped to direct Cypher lookups (`kg_query`) to bypass semantic retrieval latency.
2.  **Hybrid Fallback:** When the KG does not contain the answer nodes, the system immediately switches to our highly-efficient `text_search` tool combining a CPU-based vector database (Qdrant/FAISS) with a BM25 sparse index over paragraphs.
3.  **Low-Footprint Citations:** All retrieved facts carry `passage_ids` that are validated by a deterministic post-processor, avoiding expensive LLM-based verification passes.
