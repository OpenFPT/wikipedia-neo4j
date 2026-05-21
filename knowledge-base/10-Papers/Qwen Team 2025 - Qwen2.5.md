---
title: "Qwen2.5: Technical Report on Foundational Multilingual Models"
authors: Qwen Team
year: 2025
url: https://arxiv.org/abs/2505.09388
venue: arXiv preprint
---

tags:: [[paper]], [[local-slm]], [[vietnamese-nlp]]

# [[Qwen Team 2025 - Qwen2.5]]

## TL;DR
This technical report details the development, training, and architectural features of the Qwen2.5 multilingual model family. Qwen2.5 represents a major advancement in compact, dense models (ranging from 0.5B to 72B parameters) and mixture-of-experts (MoE) architectures, providing state-of-the-art native support for over 100 languages, complex instruction-following, and robust tool-calling.

## Method
The models are trained on a massive, highly diverse multilingual dataset spanning web documents, code, mathematics, and books. The authors utilize a modified Transformer architecture with Grouped-Query Attention (GQA) to optimize inference speed and VRAM consumption. They implement advanced byte-level BPE tokenizers and apply multi-stage reinforcement learning from human feedback (RLHF) to optimize native function calling.

## Results
*   **Academic Performance:** Qwen2.5 models consistently outperformed comparable open-source models (such as Llama 3) across multilingual benchmarks, coding tasks, and logic evaluations.
*   **Tool Calling:** Achieved high function-calling precision, approaching the performance of commercial closed-source models on structured JSON argument generation.

## Relevance
This is the architectural foundation of our local SLM:
*   **What we borrow:** The base model architecture. Our primary candidate, `Sailor2-8B-Instruct`, is built directly as a specialized SEA continuous pre-training fork of `Qwen2.5-7B`.
*   **What we adapt:** The tool-calling prompts. We adapt Qwen's standard system prompts for function-calling, translating the context and definitions to specialize the model in Vietnamese query-time orchestration.
*   **What we avoid:** Pre-training or vocabulary expansion. We utilize the pre-existing Qwen/Sailor2 tokenizer, focusing entirely on parameter-efficient downstream fine-tuning.
