---
title: "Incremental Multilingual Text2Cypher with Adapter Combination"
authors: Ozsoy et al.
year: 2026
url: https://arxiv.org/abs/2601.16097
venue: arXiv preprint
---

tags:: [[paper]], [[multilingual-text2cypher]], [[knowledge-graph]]

# [[Ozsoy et al. 2026 - Text2Cypher Adapter Fusion]]

## TL;DR
Proposes a scalable multilingual Text2Cypher approach using language-specific LoRA adapters combined via either uniform linear merging or a learned fusion MLP with dynamic gating. The goal: add new language support without re-running expensive full fine-tuning. The fusion MLP recovers ~75% of joint multilingual fine-tuning accuracy gains while requiring only a subset of data, outperforming linear merging across English, Spanish, and Turkish.

## Method
- Train independent LoRA adapters for each language (English, Spanish, Turkish) on Text2Cypher datasets.
- At inference, combine adapters via:
  - **Linear merging:** Uniform weighted average of adapter weights — simple but lossy.
  - **Fusion MLP with dynamic gating:** A lightweight MLP learns to dynamically weight adapter hidden states during generation based on input context — preserves more cross-lingual knowledge.
- Incremental expansion: adding a new language requires only training one new LoRA adapter and retraining the lightweight fusion MLP, not the base model.

## Results
- Fusion MLP recovers **~75% of accuracy gains** from full joint multilingual fine-tuning.
- Outperforms linear merging across all three languages.
- Demonstrates practical incremental language expansion with only a **smaller subset of data** compared to joint re-training.

## Relevance
This paper governs our decision on adapter strategy for Vietnamese Text2Cypher:
- **What we borrow:** The finding that a single language-specific LoRA adapter + lightweight fusion is sufficient for adding a new language. Vietnamese is not covered by the original Ozsoy & Tai 2025 (EN/ES/TR) adapter — we train one Vietnamese-specific LoRA adapter.
- **What we adapt:** We skip the fusion MLP for now (added complexity, marginal gain for single-target language) and directly merge the Vietnamese LoRA adapter into the Sailor2-8B base via QLoRA. If multi-language serving becomes a requirement, the fusion MLP is the correct upgrade path.
- **What we avoid:** Full joint multilingual retraining from scratch, which is computationally prohibitive on consumer hardware.
