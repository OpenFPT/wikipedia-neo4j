---
title: "VietMedKG: Knowledge Graph and Benchmark for Traditional Vietnamese Medicine"
authors: Trinh et al.
year: 2025
url: https://doi.org/10.1145/3744740
venue: ACM TALLIP (Vol. 24, Issue 7, 2025)
repo: https://github.com/HySonLab/VietMedKG
---

tags:: [[paper]], [[vietnamese-nlp]], [[knowledge-graph]]

# [[Trinh et al. 2025 - VietMedKG]]

## TL;DR
VietMedKG constructs a comprehensive, structured knowledge graph and query benchmark for Traditional Vietnamese Medicine (TVM). Published in ACM TALLIP 2025, the paper outlines a translation and refinement process to adapt traditional medical concepts into a localized graph schema, providing a public resource for medical QA.

## Method
The authors propose an extraction and translation framework:
*   **Graph Construction:** Leverage large traditional Chinese medicine corpora, translating and filtering relationships to match the specific flora, treatments, and terminologies of Vietnamese medicine.
*   **Benchmark Design:** Formulate QA benchmarks that trace entity relationships (e.g., matching a symptom to a herb, and a herb to a recipe) to evaluate retrieval systems.

## Results
*   **Graph Scale:** Compiled thousands of high-confidence entity-relation-entity triples mapped to localized traditional medical knowledge.
*   **Evaluation:** Benchmarked multiple baseline retrievers, demonstrating that structured graph traversals are essential for safe, factually grounded medical QA.

## Relevance
This paper directly maps to our **graph schema and entity extraction strategies**:
*   **What we borrow:** The concept of translating and localized filtering of structured entity relationships from large public datasets.
*   **What we adapt:** Translation pipeline. We adapt their localized entity translation rules to translate and normalize English Text-to-Cypher training schemas into natural Vietnamese equivalents.
*   **What we avoid:** Domain-specific medical reasoning rules, focusing our system strictly on open-domain, general-knowledge Wikipedia QA.
