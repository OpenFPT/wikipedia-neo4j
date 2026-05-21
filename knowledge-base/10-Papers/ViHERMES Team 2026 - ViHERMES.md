---
title: "ViHERMES: A Graph-Grounded Multihop Question Answering Benchmark and System for Vietnamese Healthcare Regulations"
authors: ViHERMES Team
year: 2026
url: https://arxiv.org/abs/2602.00000
venue: ACIIDS 2026
repo: https://github.com/vihermes/vihermes
---

tags:: [[paper]], [[vietnamese-nlp]], [[knowledge-graph]]

# [[ViHERMES Team 2026 - ViHERMES]]

## TL;DR
ViHERMES introduces a specialized graph-grounded multi-hop question answering benchmark and system tailored for Vietnamese healthcare regulations. The framework models complexlegal and administrative connections between regulatory documents, using a graph-aware retrieval engine to support deep relational reasoning across multiple documents.

## Method
The authors construct a specialized knowledge graph of legal nodes (such as Articles, Decrees, Clauses) and structured edges (such as `AMENDS`, `REFERENCES`, `SUPERSEDES`) in Neo4j, combined with semantic vectors in Milvus. They evaluate various multi-hop reasoning algorithms on their ability to trace logical relationships and compile correct regulatory answers.

## Results
*   **Performance:** Tracing document relationships via the knowledge graph yielded up to a 24% improvement in legal citation completeness compared to flat text search.
*   **Hallucination Rate:** Drastically reduced compliance-related hallucinations by grounding generation in verifiable legal paths.

## Relevance
This paper directly maps to our **knowledge graph schema design**:
*   **What we borrow:** The concept of explicit node-to-node relationships mapping structured references.
*   **What we adapt:** Domain target. While ViHERMES targets a highly specialized legal and healthcare regulatory context, we adapt the dual graph-vector traversal pipeline to scale across open-domain Vietnamese Wikipedia articles.
*   **What we avoid:** Heavy Milvus + Neo4j enterprise dual-hosting clusters, keeping our pipeline localized and consolidated within a single-server Neo4j database instance.
