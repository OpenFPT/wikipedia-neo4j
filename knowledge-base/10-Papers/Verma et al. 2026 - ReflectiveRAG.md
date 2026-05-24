---
title: "ReflectiveRAG: Rethinking Adaptivity in Retrieval-Augmented Generation"
authors: Akshay Verma, Swapnil Gupta, Siddharth Pillai, Prateek Sircar, Deepak Gupta
year: 2026
url: https://aclanthology.org/2026.eacl-industry.27
venue: EACL 2026 Industry Track
---

tags:: [[paper]], [[agentic-rag]], [[self-reflection]]

# [[Verma et al. 2026 - ReflectiveRAG]]

## TL;DR
ReflectiveRAG proposes a lightweight, reasoning-driven RAG architecture that addresses performance degradation when retrieved contexts contain extreme noise or redundancy. Using a small language model as a decision controller for adaptive re-retrieval, and contrastive filtering to prune redundant passages, it outperforms DeepRAG on WebQuestions, HotpotQA (distractor setting), and InternalQA — improving EM by +2.7pp and F1 by +2.5pp while adding only ~18ms of latency and reducing evidence redundancy by ~30.88%.

## Method
Two complementary mechanisms, no model fine-tuning required:
1. **Self-Reflective Retrieval (SRR):** A small LM controller evaluates whether retrieved evidence is sufficient for the current query. If not, it adaptively reformulates the query and retrieves again — without fixed schedules or costly RL policy training.
2. **Contrastive Noise Removal (NR):** Post-retrieval, embedding-based contrastive scoring removes redundant or tangential passages, enforcing semantic sparsity so the generator receives only the most pertinent content.

Evaluated on: WebQuestions, HotpotQA (distractor setting with noisy passages), InternalQA (50M Common Crawl distractors).

## Results
- **EM improvement:** +2.7 pp over DeepRAG baseline.
- **F1 improvement:** +2.5 pp.
- **Latency overhead:** ~18 ms (latency-aware design).
- **Evidence redundancy reduction:** ~30.88%.
- Key claim validated: retrieval reasoning + contrastive filtering beats large-scale policy optimization for adaptivity.

## Relevance
Directly informs our `verify_tool` and adaptive re-retrieval logic:
- **What we borrow:** The SRR "sufficiency check" concept — before generating a final answer, the orchestrator evaluates whether the retrieved graph paths / text chunks are sufficient. If not, it calls an additional retrieval tool rather than fabricating.
- **What we adapt:** Implementation. ReflectiveRAG uses a standalone small LM controller. We fold this into the ReAct loop itself — the Thought step after each tool call serves as the sufficiency evaluator.
- **What we avoid:** The Contrastive NR module as a separate pre-processing stage (latency overhead per token). We apply simple length-capped top-k retrieval instead.
