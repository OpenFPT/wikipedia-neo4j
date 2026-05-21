---
title: "DSC2025 – ViHallu Challenge: Detecting Hallucination in Vietnamese LLMs"
authors: ViHallu Challenge Organizers
year: 2026
url: https://arxiv.org/abs/2601.04711
venue: arXiv preprint
---

tags:: [[paper]], [[vietnamese-nlp]], [[self-reflection]]

# [[ViHallu Team 2026 - ViHallu]]

## TL;DR
The DSC2025 ViHallu Challenge is the first large-scale shared task for detecting hallucinations in Vietnamese LLMs. The dataset contains 10,000 annotated triplets of (context, prompt, response) across three hallucination categories — no hallucination, intrinsic (contradiction of provided context), and extrinsic (fabrication beyond context) — with three prompt types (factual, noisy, adversarial). 111 teams participated; the best system achieved macro-F1 of 84.80% vs. a baseline encoder-only score of 32.83%.

## Method
Dataset construction: systematic annotation of 10,000 (context, prompt, response) triplets partitioned into three hallucination categories. Three prompt types stress-test model robustness:
- **Factual:** Standard factual questions over the context.
- **Noisy:** Context contains irrelevant or distracting information.
- **Adversarial:** Prompts designed to elicit confabulation from the model.

Winning system approach: instruction-tuned LLMs with structured prompting and ensemble strategies, substantially outperforming generic encoder-only architectures. Key finding: **intrinsic (contradiction-based) hallucinations** remain the hardest category even for top systems.

## Results
- Best system macro-F1: **84.80%**.
- Baseline encoder-only: **32.83%** — a 52-point gap closed by instruction-tuned LLMs with structured prompting.
- **Intrinsic hallucinations** (direct contradiction of provided context) are harder than extrinsic (fabrication) — relevant because our KG-grounded verify step specifically targets intrinsic contradictions.

## Relevance
Establishes the hallucination taxonomy we use in our verification and evaluation design:
- **What we borrow:** The three-category taxonomy (no hallucination / intrinsic / extrinsic) and the three prompt types (factual / noisy / adversarial). We adopt this taxonomy to label errors in our system's outputs during evaluation.
- **What we adapt:** Detection method. ViHallu detects hallucinations post-hoc as a classification task. We *prevent* intrinsic hallucinations at generation time by running the `verify_tool` which checks the answer against KG triples before returning it to the user.
- **What we avoid:** A separate hallucination-detection model at inference time — too slow and memory-intensive for a local pipeline.
