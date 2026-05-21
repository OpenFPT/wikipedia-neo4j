---
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
year: 2023
url: https://arxiv.org/pdf/2305.18290
venue: NeurIPS 2023
---

tags:: [[paper]], [[preference-optimization]], [[reinforcement-learning]], [[alignment]]

# [[Rafailov et al. 2023 - DPO]]

## TL;DR
Direct Preference Optimization (DPO) is an elegant algorithm designed to align large language models with human preferences without the need for reinforcement learning. The authors show that the standard RLHF objective has a closed-form solution, enabling direct policy training using a simple binary cross-entropy loss over pairwise preference data. This eliminates the instabilities and resource overhead associated with standard PPO pipelines.

## Method
DPO reformulates the RLHF objective (maximizing a reward function with a KL-divergence constraint relative to a reference policy) by expressing the latent reward function analytically in terms of the policy itself. This mathematical substitution allows the policy to be optimized directly on preference datasets containing chosen and rejected response pairs. The optimization uses a simple binary classification loss that increases the log probability of preferred responses and decreases the log probability of rejected responses, regulated by a scaling factor beta.

## Results
*   **Stability & Implementation:** Bypassed the separate reward modeling and PPO actor-critic training stages, completely eliminating reward-hacking instabilities.
*   **Sample Efficiency:** Matched or exceeded PPO performance in tasks like summarization and conversational dialogue while being significantly faster to train.
*   **Hyperparameter Tuning:** Demonstrated that a simple KL penalty parameter beta (between 0.1 and 0.5) is highly stable and robust to tuning variations.

## Relevance
*   **What we borrow:** The pairwise DPO binary loss to train our adapter models on preference datasets.
*   **What we adapt:** ReAct loop formatting alignment. We will use DPO as a secondary alignment phase to force our local SLM to output strict JSON formatting blocks matching the ReAct agent standard (forcing structured tool parameters in `Action: {"tool": "...", "args": {...}}` with a cap of 6 steps).
    *   **Chosen ($y_w$):** Clean JSON tool calls containing exactly the required arguments without conversational preamble.
    *   **Rejected ($y_l$):** Queries wrapped in conversational fluff (e.g., *"Sure, let me check the database for you: ```json ..."*) or invalid schema property structures.
*   **What we avoid:** Complex reinforcement learning pipelines (PPO), which require active token sampling during optimization and are computationally impractical for our resource-constrained 1-GPU developer setup.

```mermaid
graph TD
    subgraph Traditional RLHF (PPO)
        DataRL[Preference Data] --> RewardM[Train Reward Model]
        RewardM --> PPO[Run PPO Actor-Critic Loop]
        PPO --> VSpikes[Susceptible to Reward Hacking & VRAM spikes]
    end

    subgraph Direct Preference Optimization (DPO)
        DataDPO[Preference Data] --> Loss[Direct DPO Binary Loss]
        Loss --> Stable[Stable Classification Gradient Update]
    end
```
