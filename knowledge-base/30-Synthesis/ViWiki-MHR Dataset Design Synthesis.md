# ViWiki-MHR Dataset Design Synthesis

This document outlines the design, schema, taxonomic classification, construction methodology, and quality control pipeline for the **ViWiki-MHR** dataset.

---

## 1. Dataset Overview

ViWiki-MHR is a multi-hop, retrieval-required, open-license Vietnamese Question Answering dataset consisting of **~36,000 examples** grounded on a pinned snapshot of Vietnamese Wikipedia. 

### Core Design Philosophy
1.  **Retrieval-Required (Corpus-Scale):** Unlike reading comprehension datasets (like UIT-ViQuAD or ViMQA) that provide the model with "gold context" passages, ViWiki-MHR requires models to query a fixed 220K+ paragraph Wikipedia corpus. Questions map to `gold_passage_ids` rather than gold text blocks, evaluating retrieval and answering jointly.
2.  **Open Science & Reproducibility:** Released under a permissive **CC-BY-SA** license on Hugging Face. This resolves the distribution gates of prior Vietnamese multi-hop resources like ViMQA (whose EULA requires bilateral correspondence and restricts downstream derivative reuse).
3.  **Adversarial Rigor:** Integrates single-hop adversarial unanswerables from UIT-ViQuAD 2.0 alongside a novel, custom-synthesized **Multi-hop Adversarial Unanswerable** subset to measure and mitigate model hallucinations.

---

## 2. Dataset Schema & Metadata Structure

Each QA pair in the dataset is represented as a structured JSON object containing reasoning tags, provenance data, Cypher queries, and intermediate sub-question annotations:

```json
{
  "id": "viwiki_mhr_2026_004928",
  "question": "Câu lạc bộ mà John O'Shea gia nhập khi anh ấy 17 tuổi nằm ở thành phố nào?",
  "answer": "Manchester",
  "source": "synthetic-multihop",
  "num_hops": 2,
  "answerable": true,
  "reasoning_type": "bridge",
  "gold_passage_ids": [
    "wp_passage_john_oshea_01",
    "wp_passage_manchester_united_00"
  ],
  "cypher_query": "MATCH (p:Person {name: 'John O\'Shea'})-[:GIA_NHẬP {tuổi: 17}]->(c:CâuLạcBộ)-[:TRỤ_SỞ]->(t:ThànhPhố) RETURN t.name AS answer",
  "decomposition_annotations": [
    {
      "sub_question": "John O'Shea gia nhập câu lạc bộ nào khi 17 tuổi?",
      "sub_answer": "Manchester United",
      "sub_gold_passage_id": "wp_passage_john_oshea_01"
    },
    {
      "sub_question": "Câu lạc bộ Manchester United nằm ở thành phố nào?",
      "sub_answer": "Manchester",
      "sub_gold_passage_id": "wp_passage_manchester_united_00"
    }
  ]
}
```

### Schema Parameters
*   `id` *(string)*: Unique dataset entry key.
*   `question` *(string)*: Natural language Vietnamese question.
*   `answer` *(string)*: Ground truth answer text (span extraction, name, date, or "I don't know").
*   `source` *(string)*: `uit-viquad` (reformatted baseline) or `synthetic-multihop` (our generation).
*   `num_hops` *(integer)*: Number of relational reasoning links required (1, 2, or 3+).
*   `answerable` *(boolean)*: True if the answer is derivable from the corpus; False for adversarial unanswerables.
*   `reasoning_type` *(string)*: Classification tag (e.g., `lookup`, `bridge`, `comparison`, `intersection`, `temporal`, `fan-out`).
*   `gold_passage_ids` *(array of strings)*: Unique identifiers of the Wikipedia paragraphs containing the supporting facts.
*   `cypher_query` *(string)*: For multi-hop synthetic queries, the gold schema-compliant Cypher path used to extract the entities.
*   `decomposition_annotations` *(array of objects)*: Intermediate sub-questions and answers mapped to their respective supporting passage IDs, serving as supervision for question decomposition training.

---

## 3. Dataset Composition & Provenance

We build ViWiki-MHR by combining reformatted and cleaned open subsets from **UIT-ViQuAD 2.0** with newly synthesized Knowledge Graph (KG) multi-hop structures:

```
                  ┌──────────────────────────────────────────────┐
                  │          ViWiki-MHR Corpus (~36K)            │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     ┌───────────────────────┐                       ┌───────────────────────┐
     │  UIT-ViQuAD 2.0 (28K) │                       │ Synthetic KG-QA (8K)  │
     └───────────┬───────────┘                       └───────────┬───────────┘
                 │                                               │
        ┌────────┴────────┐                             ┌────────┴────────┐
        ▼                 ▼                             ▼                 ▼
 ┌─────────────┐   ┌─────────────┐               ┌─────────────┐   ┌─────────────┐
 │ 1-Hop       │   │ 1-Hop       │               │ Multi-Hop   │   │ Multi-Hop   │
 │ Answerable  │   │ Adversarial │               │ Answerable  │   │ Adversarial │
 │ (23K)       │   │ Unanswerable│               │ (7K)        │   │ Unanswerable│
 │             │   │ (5K Sub)    │               │             │   │ (1K)        │
 └─────────────┘   └─────────────┘               └─────────────┘   └─────────────┘
```

---

## 4. Reasoning Taxonomy

We adapt and extend reasoning taxonomies from HotpotQA, MuSiQue, and ViMQA to establish a robust, query-focused classification framework:

### 1. Lookup / Single-Hop (~28K)
*   *Definition:* Requires locating a single fact or attribute of a specific entity.
*   *Source:* Adapted from UIT-ViQuAD 2.0.

### 2. Bridge Reasoning (~5K)
*   *Definition:* The model must identify a bridging entity ($B$) connected to $A$, then retrieve an attribute of $B$ to locate $C$.
*   *KG Traversal:* `(A)-[:R1]->(B)-[:R2]->(C)`

### 3. Comparison (~1K)
*   *Definition:* Requires retrieving attributes for two distinct entities and performing an comparative evaluation (e.g., comparing heights, lengths, or birth years).
*   *KG Traversal:* `(A)-[:ATTR1]->(X)` and `(B)-[:ATTR1]->(Y)` followed by a comparison step.

### 4. Intersection (~1K)
*   *Definition:* The query searches for an entity that satisfies multiple distinct constraints (e.g., "Which Vietnamese writer was born in Year X AND composed literary work Y?").
*   *KG Traversal:* `(Writer)<-[:SÁNG_TÁC]-(Y)` and `(Writer {năm_sinh: X})`

### 5. Temporal Subset (~2K)
*   *Definition:* The answer is dependent on specific dates, periods, historical epochs, or sequences (e.g., "Who was prime minister of Vietnam during historical event X?").
*   *KG Traversal:* Node properties containing date ranges or timestamped relationship attributes.

### 6. Fan-out Subset (~1K)
*   *Definition:* Requires enumerating multiple matching entities rather than extracting a single span (e.g., "List all provinces bordering Laos with population over 500K").
*   *KG Traversal:* Aggregating nodes matching complex filter predicates.

---

## 5. Adversarial Multi-Hop Construction

While UIT-ViQuAD 2.0 provides high-quality single-hop adversarial unanswerables, **multi-hop unanswerability is a major benchmark gap in Vietnamese**. We generate ~1,000 multi-hop unanswerables programmatically:

```
Step 1: Walk Valid Path ──► (A) ─[:R1]─► (B) ─[:R2]─► (C)
                                                       │
                                                       ▼
Step 2: Swap Final Entity ──► (A) ─[:R1]─► (B) ─[:R2]─► [Fabricated: X]
                                                       │
                                                       ▼
Step 3: Natural VN Rewrite ──► "Who was C of B linked to A?" (Coherent syntax)
                                                       │
                                                       ▼
Step 4: Verification Check ──► Run Text2Cypher check against KG.
                               Confirm query returns Empty Result.
                               Verify passages do NOT contain answer.
```

If a frontier verification model attempts the question and confidently invents a response, we retain the record as a **high-quality adversarial multi-hop unanswerable** that exposes local SLM hallucination.

---

## 6. Dataset Quality Control Pipeline

We apply a strict three-stage validation pipeline to guarantee the semantic validity, grounding, and syntax quality of all synthesized multi-hop elements:

```
  ┌────────────────────────────────────────────────────────┐
  │              Synthesized KG Walk Pairs                 │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │           Stage 1: Grounding & Match Filter            │
  │  - Programmatic check: Is answer string fully present  │
  │    within the referenced Wikipedia paragraphs?         │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │             Stage 2: NLI Entailment Filter             │
  │  - Model-based validation: Does a pre-trained NLI      │
  │    model confirm that passages entail the answer?      │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │        Stage 3: Independent Well-Formed Filter         │
  │  - Independent LLM verifier parses the query to check   │
  │    for semantic clarity and syntactic coherence.        │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │               Human-Verified Test Split                │
  │  - 1,000 to 2,000 samples spot-checked and cleaned     │
  │    manually to ensure 100% ground truth accuracy.      │
  └────────────────────────────────────────────────────────┘
```

---

## 7. Strategic Positioning in Related Work

When writing the related work and methods sections of the paper, we position **ViWiki-MHR** neutrally and factually:

*   **Relationship to UIT-ViQuAD:** We acknowledge UIT-ViQuAD 2.0 as the premier single-hop reading comprehension corpus in Vietnamese, citing their VLSP 2021 work. We build upon their effort by extracting their answerable and adversarial sets, unifying them into our modular RAG schema, and expanding the pipeline into multi-hop domains.
*   **Relationship to ViMQA:** We cite ViMQA as the closest prior Vietnamese multi-hop reading comprehension baseline, highlighting its structural taxonomy. However, we state factually that:
    > *"Existing Vietnamese multi-hop QA resources are restricted by user agreements requiring bilateral correspondence with the authors, which limits downstream reuse and hampers reproducible open research. In this work, we present ViWiki-MHR, a fully independent 36K-example benchmark released under a CC-BY-SA license with no access gates, alongside the first Vietnamese multi-hop adversarial unanswerable subset for systematic hallucination measurement."*
*   **Headline Evaluation:** Because we do not have license rights or access to ViMQA data, we perform zero training or evaluation on their splits. Our benchmark tables report system accuracies (base SLM prompting, vector-only RAG, BM25-only RAG, and our fine-tuned local hybrid KG-QA orchestrator) **entirely on our own open ViWiki-MHR test split**, establishing a reproducible baseline for the Vietnamese NLP community.
