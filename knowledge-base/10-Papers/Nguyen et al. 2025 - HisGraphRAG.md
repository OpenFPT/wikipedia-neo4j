---
title: "HisGraphRAG: A Graph-Based Retrieval-Augmented Generation System for Vietnamese Historical Question Answering"
authors: Hoang Thanh Nguyen, Tung Le, Huy Tien Nguyen
year: 2025
url: https://aclanthology.org/2025.paclic-1.0
venue: PACLIC 39 (2025)
---

tags:: [[paper]], [[vietnamese-nlp]], [[graph-rag]]

# [[Nguyen et al. 2025 - HisGraphRAG]]

## TL;DR
HisGraphRAG is a specialized Graph-based Retrieval-Augmented Generation system designed for answering historical questions in Vietnamese. The framework constructs a custom historical knowledge graph and uses entity matching to retrieve sub-graphs for context generation, showing improvements over traditional vector-only RAG pipelines on domain-specific historical facts.

## Method
The authors construct a Knowledge Graph from historical Vietnamese texts by extracting historical figures, dynasties, events, and locations. They utilize an entity-linking module to map user questions to target graph nodes. When a query is matched, the system extracts the neighboring entity-relation-entity triples and raw source text blocks, using an LLM to synthesize a grounded response.

## Results
*   **Evaluation:** Benchmarked on a custom historical QA dataset in Vietnamese.
*   **Accuracy:** Achieved up to a 15% absolute improvement in answer precision and factual completeness over standard document-chunk vector RAG.
*   **Context Reduction:** Reduced input context token length by over 40% by extracting targeted graph triples instead of retrieving massive flat text documents.

## Relevance
This is the closest regional baseline for GraphRAG in Vietnamese:
*   **What we borrow:** The concept of entity-grounded sub-graph extraction for context pruning. Matching entities to node properties and feeding adjacent relationships to the model is the right strategy.
*   **What we adapt:** Domain scaling. While HisGraphRAG targets a highly specialized, closed historical subset, we scale the pipeline to ingest a broad, open-domain corpus (Vietnamese Wikipedia) using Neo4j.
*   **What we avoid:** Heavy dependency on manual entity linking rules. We implement a dynamically fine-tuned Text-to-Cypher translation model to compile natural language questions directly into query scripts, enabling programmatic traversal of complex, multi-hop pathways.
