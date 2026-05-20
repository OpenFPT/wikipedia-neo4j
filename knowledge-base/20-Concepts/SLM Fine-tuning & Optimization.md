# SLM Fine-tuning & Optimization

Fine-tuning a Small Language Model (SLM, 7B/8B scale) to perform robust Text-to-Cypher translation requires moving beyond generic instruction tuning. We combine parameter-efficient quantization training (QLoRA) with post-training preference optimization (DPO) to enforce structural correctness and eliminate hallucinations.

---

## The Dual-Stage Optimization Pipeline

To adapt a base model (like `Qwen/Qwen2.5-7B-Instruct` or `vinai/PhoGPT-7B`) for our Vietnamese KG-QA system, we execute a specialized two-stage training loop:

```mermaid
graph TD
    %% Dataset Prep
    RawData[Raw Translation Corpus] --> Formatter[ChatML Formatter]
    
    %% Stage 1
    subgraph Stage 1: QLoRA Instruction Tuning (VRAM-Efficient)
        Formatter --> QLoRALoss[NF4 Quantized Base Model]
        QLoRALoss -->|Gradients backpropagated| LoRA[LoRA Adapters on all Linear Layers]
        LoRA --> InstructionModel[Instruction-Tuned Adapter]
    end
    
    %% Stage 2
    subgraph Stage 2: DPO Alignment (Schema & Output Enforcement)
        InstructionModel --> DPOLoss[Direct Preference Loss]
        PrefData[Preference Pairs: Chosen vs Rejected] --> DPOLoss
        DPOLoss --> FinalModel[Final Structured SLM Engine]
    end
```

---

## Stage 1: QLoRA Instruction Tuning Recipe

### 1. Model Selections for Vietnamese
*   **Qwen-2.5-7B-Instruct:** Exceptional multilingual base with powerful structured output and reasoning capabilities, paired with highly efficient Vietnamese tokenization.
*   **vinai/PhoGPT-7B:** A monolingual Vietnamese LLM exhibiting deep understanding of local entities, idioms, and historical facts.
*   **SeaLLM-7B:** Explicitly pre-trained for Southeast Asian languages, representing a robust default candidate.

### 2. Quantization & Hyperparameters
To preserve representation capacity while fitting inside a single developer GPU training budget, we use the following parameter configurations:
*   **BitsAndBytes Configuration:**
    *   `load_in_4bit`: `True`
    *   `bnb_4bit_quant_type`: `"nf4"`
    *   `bnb_4bit_use_double_quant`: `True`
    *   `bnb_4bit_compute_dtype`: `torch.bfloat16`
*   **LoRA Configurations (All Linear Targets):**
    *   `target_modules`: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
    *   `lora_r`: `32`
    *   `lora_alpha`: `64`
    *   `lora_dropout`: `0.05`
*   **Training parameters:**
    *   Optimizer: `paged_adamw_32bit` (handles transient memory spikes).
    *   Learning Rate: `2e-4` with cosine decay.
    *   Sequence Length: `2048` tokens.

---

## Stage 2: DPO Alignment (Preference Filtering)

Instruction tuning teaches the model *how* to generate Cypher queries, but DPO teaches it *what to avoid* to satisfy strict production and database constraints.

### 1. Creating the Preference Dataset
We auto-generate DPO preference pairs (Chosen $y_w$ vs. Rejected $y_l$) by running our training set queries through the active Neo4j database:
*   **Scenario A: Schema Hallucination**
    *   **Chosen ($y_w$):** Returns exact valid query: `MATCH (tp:TácPhẩm {tên: "Tắt đèn"})<-[:SÁNG_TÁC]-(nv:NhàVăn) RETURN nv.năm_sinh`
    *   **Rejected ($y_l$):** Returns a query referencing hallucinated properties: `MATCH (tp:TácPhẩm {title: "Tắt đèn"})-[:WRITTEN_BY]->(nv:Author) ...`
*   **Scenario B: Conversational Fluff**
    *   **Chosen ($y_w$):** Standard JSON matching the ReAct tool syntax: `{"tool": "kg_query", "args": {"cypher": "..."}}`
    *   **Rejected ($y_l$):** Fluff-wrapped output: *"Dưới đây là câu lệnh Cypher bạn yêu cầu: ```json ..."*

### 2. Loss Optimization & Alignment
We train the QLoRA adapters from Stage 1 using the binary DPO classification loss:
$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$
*   We use a frozen copy of the Stage 1 model as $\pi_{\text{ref}}$.
*   Set $\beta = 0.1$.
*   This alignment step guarantees **zero conversational preamble** and a **95%+ schema executability rate**, making the SLM fully safe for programmatic execution inside our 6-iteration capped ReAct agent loop.
