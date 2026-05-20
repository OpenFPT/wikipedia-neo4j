---
title: "QLoRA: Efficient Finetuning of Quantized LLMs"
authors: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
year: 2023
url: https://arxiv.org/pdf/2305.14314
venue: NeurIPS 2023
---

tags:: [[paper]], [[slm-finetuning]], [[quantization]], [[parameter-efficiency]]

# [[Dettmers et al. 2023 - QLoRA]]

## TL;DR
QLoRA is an efficient fine-tuning approach that enables training large language models on consumer-grade hardware without sacrificing performance. It allows backpropagating gradients through a frozen, 4-bit quantized base model into active Low-Rank Adapters (LoRA). This technique reduces memory usage, making the fine-tuning of 7B-65B parameter models highly accessible.

## Method
The authors introduced three core innovations to achieve high VRAM efficiency: 4-bit NormalFloat (NF4), Double Quantization (DQ), and Paged Optimizers. NF4 is an information-theoretically optimal quantile quantization data type designed for normally distributed weights. Double Quantization quantizes the quantization constants themselves, saving further memory. Paged Optimizers leverage NVIDIA Unified Memory to page memory spikes to CPU RAM during gradient checkpointing, preventing out-of-memory errors during long sequence processing.

## Results
*   **VRAM Optimization:** Reduced the memory footprint of a 7B parameter model to 6-8 GB during fine-tuning (allowing it to train easily on a single 16GB-24GB consumer GPU).
*   **Bits Saved:** Double Quantization saved an average of 0.37 bits per parameter.
*   **Performance Parity:** Matched 16-bit fully fine-tuned baselines across standard benchmarks (Vicuna, Alpaca) while training the active parameters of LoRA.

## Relevance
This is the foundational recipe for fine-tuning our specialized local Vietnamese Small Language Model (SLM).
*   **What we borrow:** The 4-bit NormalFloat (NF4) quantization configurations, double quantization, and target adapter modules (targeting all linear layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
*   **What we adapt:** Hardware targeting. We will use QLoRA to fine-tune our adapter on a single 24GB developer GPU. At inference time, this allows us to compile the entire 7B base model + adapter into 4-bit quantization, enabling it to run smoothly on our target consumer-grade spec: **16GB system RAM, 8GB VRAM (RTX 3060/4060 class)** or 16GB Apple Silicon.
*   **What we avoid:** Quantizing the embedder. The embedding model (e.g., `bge-m3` or similar) runs directly on CPU without quantization to preserve high dense search recall.

| Fine-Tuning Type | Base Model Precision | Optimizer States | Adapter Precision | VRAM Needed |
| :--- | :--- | :--- | :--- | :--- |
| **Full Finetuning** | FP16 / BF16 (14 GB) | FP32 AdamW (28 GB) | N/A | **~48+ GB** |
| **Standard LoRA** | FP16 (14 GB) | FP32 AdamW (small) | FP32/FP16 (LoRA) | **~18 - 24 GB** |
| **QLoRA** | **NF4 (3.5 GB)** | **Paged AdamW (small)** | **BF16 (LoRA)** | **~6 - 8 GB** |
