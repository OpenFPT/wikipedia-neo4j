---
title: "TinyLLM: Evaluation and Optimization of Small Language Models for Agentic Tasks on Edge Devices"
authors: TinyLLM Authors
year: 2025
url: https://arxiv.org/abs/2511.22138
venue: arXiv preprint
---

tags:: [[paper]], [[local-slm]], [[tool-calling]]

# [[TinyLLM Team 2025 - TinyLLM]]

## TL;DR
TinyLLM investigates the effectiveness of Small Language Models (SLMs) for agentic function/tool/API calling tasks on edge devices without cloud infrastructure. It evaluates SLMs using the Berkeley Function Calling Leaderboard (BFCL) framework and describes a range of optimization strategies: SFT, PEFT, RL-based optimization, DPO preference alignment, and hybrid combinations. The key finding is that medium-sized models (1–3B) significantly outperform ultra-compact models (<1B) with hybrid optimization reaching 65.74% overall BFCL accuracy.

## Method
Models evaluated include TinyAgent, TinyLlama, Qwen, and xLAM across BFCL categories: simple, multiple, parallel, parallel-multiple, and relevance detection — in both live and non-live settings and multi-turn evaluation. A DPO training pipeline is constructed from AgentBank data (e.g., ALFRED), converting SFT data to chosen/rejected pairs by using TinyLlama responses as rejected outputs and manually validated gold outputs as chosen. Optimization strategies compared: pure SFT, PEFT (LoRA), RL (PPO-style), DPO, and hybrid SFT+DPO.

## Results
- Medium-sized SLMs (1–3B) achieve **up to 65.74% overall BFCL accuracy** and **55.62% multi-turn accuracy** with hybrid optimization.
- Hybrid SFT+DPO consistently outperforms any single strategy.
- Ultra-compact models (<1B) are insufficient for reliable agentic tasks even with optimization.
- Highlights that privacy-preserving, low-latency autonomous agents on edge are practical with the right training strategy.

## Relevance
This provides empirical support for our SFT+DPO training recipe and hardware target:
- **What we borrow:** The hybrid SFT+DPO optimization strategy. We apply SFT on successful tool-calling trajectories first, then DPO on chosen (valid Cypher / correct JSON) vs. rejected (syntax-broken / hallucinated) pairs.
- **What we adapt:** Model scale. We target 7–8B models rather than 1–3B, giving us a stronger starting capability for multilingual Vietnamese reasoning while still fitting in our 8–16GB VRAM target.
- **What we avoid:** RL-based optimization (PPO), which requires a separate reward model and is computationally too heavy for our hardware constraints.
