---
title: "Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs"
authors: Sailor2 Team (Sea AI Lab)
year: 2025
url: https://arxiv.org/abs/2502.12982
license: Apache 2.0
repo: https://huggingface.co/sail/Sailor2-8B-Chat
venue: arXiv preprint
---

tags:: [[paper]], [[vietnamese-nlp]], [[local-slm]]

# [[Sailor2 Team 2025 - Sailor2]]

## TL;DR
Sailor2 is a family of multilingual SLMs (1B, 8B, 20B) purpose-built for South-East Asian languages, supporting 13 SEA languages while retaining strong Chinese and English proficiency. Built on Qwen2.5, it undergoes continuous pre-training on 500B tokens (400B SEA-specific, 100B replay). Sailor2-20B achieves a 50–50 win rate against GPT-4o across SEA languages. Released under Apache 2.0.

## Method
Built on the **Qwen2.5** base architecture with continuous pre-training using a carefully curated SEA multilingual corpus:
- **400B SEA-specific tokens** from web crawls, governmental documents, books, and parallel translation corpora across 13 SEA languages (Vietnamese, Thai, Indonesian, Malay, Tagalog, Burmese, Khmer, Lao, Tamil, Cebuano, Javanese, Sundanese, Ilocano).
- **100B replay tokens** from Chinese and English to prevent catastrophic forgetting.
- Post-training covers SFT and preference optimization aligned for multilingual instruct-following tasks.
- Comprehensive cookbook released covering data curation, pre-training, post-training, model customization, and evaluation.

## Results
- **Sailor2-20B:** 50–50 win rate against GPT-4o across SEA languages.
- **Sailor2-8B:** Strong Vietnamese performance on regional academic benchmarks, substantially outperforming Llama 3 and Mistral 7B equivalents on Vietnamese-specific tasks.
- Quantized 4-bit variants retain >95% of full-precision performance, fitting within 6GB VRAM.

## Relevance
This is our primary local SLM base model for fine-tuning:
- **What we borrow:** `Sailor2-8B` base weights — pre-trained on 400B SEA tokens including Vietnamese, giving us a strong Vietnamese linguistic prior before any task-specific fine-tuning.
- **What we adapt:** Post-training. Sailor2's SFT targets general multilingual instruction-following. We apply QLoRA on top for a narrow, specialized task: Vietnamese Text-to-Cypher generation and ReAct 4-tool orchestration.
- **What we avoid:** Pre-training from scratch or vocabulary extension — we fully rely on the existing Qwen2.5/Sailor2 tokenizer and vocabulary, which already handles Vietnamese diacritics correctly.
