---
title: "ViWiQA: An Efficient Open-Domain Question Answering System for Vietnamese Wikipedia"
authors: ViWiQA Team
year: 2023
url: https://doi.org/10.1016/j.ipm.2023.00000
venue: Information Processing & Management (2023)
---

tags:: [[paper]], [[vietnamese-nlp]], [[hybrid-retrieval]]

# [[ViWiQA Team 2023 - ViWiQA]]

## TL;DR
ViWiQA introduces an efficient, end-to-end open-domain question answering framework designed specifically for Vietnamese Wikipedia. The system addresses the limitations of sparse retrieval in low-resource settings, proposing hybrid retriever designs that significantly improve the factual precision of extracted passage contexts.

## Method
The authors construct a dual-path indexing and retrieval pipeline:
*   **Retriever Module:** Combines sparse Lucene-BM25 indices with dense passage retrieval (DPR) optimized for Vietnamese syllable structure and diacritics.
*   **Reader Module:** Uses a fine-tuned Transformer model to locate and extract precise answer substrings from the combined retrieved contexts.

## Results
*   **Performance Gains:** The hybrid sparse-dense retriever significantly outperformed traditional BM25 baselines on both single-hop and multi-hop question sets.
*   **F1 Score:** Demonstrated substantial gains in Answer F1 and context precision, establishing a high-quality baseline for Vietnamese open-domain MRC.

## Relevance
This paper directly maps to our **hybrid text retrieval baseline**:
*   **What we borrow:** The sparse-dense hybrid retrieval design. We adopt the combined use of sparse (BM25) and dense indices in our fallback text retrieval tool.
*   **What we adapt:** Reader module. While ViWiQA uses a simple extractive Reader to pull substrings, we replace this with our fine-tuned local SLM orchestrator which synthesizes complete natural language answers with explicit citation tags.
*   **What we avoid:** Complex, separately hosted DPR neural networks that consume excessive GPU memory.
