---
title: "Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning"
authors: Birkholm et al.
year: 2025
url: https://arxiv.org/abs/2512.15943
venue: arXiv preprint
---

tags:: [[paper]], [[local-slm]], [[tool-calling]]

# [[Birkholm et al. 2025 - Efficient Agentic Tool Calling]]

## TL;DR
Investigates replacing LLM-driven workflows with domain-adapted SLMs for enterprise agentic tasks (document summarization, query answering, structured data interpretation). Fine-tuning OPT-350M using Hugging Face TRL SFT Trainer on a representative task corpus, the resulting SLM achieves 77.55% pass rate on ToolBench, significantly outperforming ChatGPT-CoT (26.00%), ToolLLaMA-DFS (30.18%), and ToolLLaMA-CoT (16.27%).

## Method
The authors fine-tune `facebook/opt-350m` (one epoch) using the Hugging Face TRL SFT Trainer on a curated domain task corpus covering: document summarization, QA, and structured data interpretation — tasks representative of enterprise LLM workloads. The evaluation metric is ToolBench pass rate, measuring whether the model's tool calls produce correct, executable results.

## Results
- Fine-tuned OPT-350M: **77.55% ToolBench pass rate**.
- ChatGPT-CoT baseline: **26.00%** — a massive 51.55% gap.
- ToolLLaMA-DFS: **30.18%**, ToolLLaMA-CoT: **16.27%**.
- Demonstrates that even 350M parameter models, when properly fine-tuned on targeted task data, substantially outperform much larger general-purpose models on specific tool execution tasks.

## Relevance
Provides evidence that targeted SFT on task-specific data is the right approach for our tool-calling alignment:
- **What we borrow:** The principle of targeted SFT over general instruction tuning. We do not attempt to improve general reasoning — we specifically fine-tune our Sailor2-8B on Vietnamese Text-to-Cypher trajectories and ReAct tool-call sequences.
- **What we adapt:** Model scale (350M → 8B) and task (generic enterprise tools → `kg_query`, `text_search`, `get_passage`, `verify_tool`). Our domain is Vietnamese Wikipedia, not English enterprise documents.
- **What we avoid:** Attempting to compete with GPT-4o on general tasks. Our system is purpose-built for one job: Vietnamese Wikipedia multi-hop QA via a 4-tool ReAct loop.
