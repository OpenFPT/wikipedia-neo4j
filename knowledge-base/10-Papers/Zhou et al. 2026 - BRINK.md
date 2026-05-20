---
title: "What Breaks Knowledge Graph based RAG? Benchmarking and Empirical Insights into Reasoning under Incomplete Knowledge"
authors: Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang, Hongkuan Zhou, Yuan He, Jiaoyan Chen, Steffen Staab, Evgeny Kharlamov
year: 2026
url: https://arxiv.org/abs/2508.08344
venue: EACL 2026
repo: https://github.com/boschresearch/brink
---

tags:: [[paper]], [[knowledge-graph]], [[kg-incompleteness]]

# [[Zhou et al. 2026 - BRINK]]

## TL;DR
BRINK (Benchmark for Reasoning under Incomplete Knowledge) identifies a critical flaw in existing KG-RAG evaluation: most benchmarks contain questions answerable by directly retrieving a single existing triple, so it is impossible to tell whether models actually reason or simply look up memorized facts. BRINK constructs benchmarks specifically requiring multi-hop inference over *missing* triples — revealing that current KG-RAG methods have severely limited reasoning ability under knowledge incompleteness and often fall back on internal memorization.

## Method
The authors develop a general benchmark construction method:
1. Mine high-confidence logical rules from the target KG (e.g. FB15k-237, Wikidata5m).
2. Identify triples that are *inferable* via those rules but not explicitly stored.
3. Remove the inferable target triple from the graph while preserving the supporting alternative paths needed for logical derivation.
4. Formulate questions that require traversing the now-absent triple, forcing models to reason across remaining evidence rather than retrieve directly.

Three benchmarks are released: **BRINK-family**, **BRINK-FB15k-237**, and **BRINK-Wikidata5m**, hosted on Hugging Face. Code is at `github.com/boschresearch/brink`.

## Results
- Current KG-RAG methods show severely **limited reasoning under missing knowledge**, frequently hallucinating confident answers for questions whose supporting triples have been removed.
- Models exhibit **varying degrees of generalization** depending on design — retrieval-augmented models outperform pure parametric models but still rely heavily on internal memorization.
- Inconsistent evaluation metrics and lenient answer matching in prior benchmarks were found to obscure meaningful comparison between systems.

## Relevance
This paper directly justifies our **adversarial broken-link unanswerable subset** in ViWiki-MHR:
- **What we borrow:** The core methodology — walk a valid n-hop path in the Vietnamese Wikipedia KG, remove one key relation triple, then formulate a question that requires that exact traversal. This is our programmatic unanswerable generation algorithm.
- **What we adapt:** Scale and language. BRINK targets English KGs (FB15k-237, Wikidata5m). We apply an equivalent procedure to our Vietnamese Wikipedia Neo4j graph.
- **What we avoid:** The rule-mining pipeline using logical rule learners (e.g. AnyBURL). For construction efficiency, we use simpler path-walking with random edge deletion since our goal is adversarial coverage, not logical completeness.
