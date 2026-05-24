# Dataset Collections

This document catalogs the metadata, licenses, corpus sizes, and structural roles of the five primary public and custom datasets supporting our Vietnamese KG-QA system.

---

## 1. ViWiki-MHR (Our Core Dataset)
*   **Role:** The primary multi-hop and adversarial benchmark for system evaluation.
*   **Size:** ~36,000 question-answer examples.
*   **Licensing:** Permissive **CC-BY-SA 4.0** (Completely Open).
*   **Format:** JSON records mapping natural language queries to `gold_passage_ids`, Cypher query strings, and intermediate reasoning tags.
*   **Origin:** Independently synthesized via template-based walks on a local Neo4j graph of Vietnamese Wikipedia, validated by a frontier model filter, and spot-checked by human annotators.
*   **Strategic Value:** Serves as the first completely open, reproducible multi-hop Vietnamese retrieval benchmark, resolving the distribution barriers of previous gated datasets.

---

## 2. UIT-ViQuAD 2.0
*   **Role:** The baseline source for single-hop answerable and unanswerable QA.
*   **Size:** 35,990 question-answer pairs derived from Vietnamese Wikipedia.
*   **Licensing:** Public Research License (VLSP 2021).
*   **Hugging Face Source:** `uitnlp/vietnamese_viquad` (SQuAD v2.0-style format).
*   **Format:** Paragraph-bound MRC contexts with start-character answer indices.
*   **Integration:** We reformat **~23K answerable** and **~5K adversarial unanswerable** single-hop examples from this corpus, unifying them into the `uit-viquad` source tag in our ViWiki-MHR schema to evaluate single-hop retrieval baseline performance.

---

## 3. Neo4j Text2Cypher
*   **Role:** The baseline corpus for alignment fine-tuning (Text-to-Cypher translation).
*   **Size:** 40,000 Cypher generation training pairs.
*   **Licensing:** Permissive **Apache 2.0**.
*   **Hugging Face Source:** `neo4j/text2cypher-2025v1`.
*   **Format:** Schema definitions, English natural language prompts, and corresponding Cypher execution strings.
*   **Integration:** Serves as the structural framework for our primary QLoRA fine-tuning adapter. We translate and adapt a target subset (~10–15K pairs) into Vietnamese to compile our schema-aligned training corpus.

---

## 4. VIMQA (Le et al. 2022)
*   **Role:** Out-of-scope structural benchmark.
*   **Size:** 10,047 human-generated multi-hop Wikipedia QA pairs.
*   **Licensing:** **EULA-Gated** (Requires bilateral correspondence with authors; restricts derivative distribution and commercial application).
*   **Strategic Stance:** Purged entirely from our training and evaluation datasets to prevent compliance violations. We cite their EACL/LREC 2022 publication strictly to adapt their structural multi-hop taxonomy (Bridge, Comparison, and Intersection reasoning).

---

## 5. Vietnamese_RAG / ViNewsQA
*   **Role:** Supplementary/Out-of-scope corpus.
*   **Size:** 8K+ entries (Vietnamese_RAG) and 22,057 pairs (ViNewsQA).
*   **Licensing:** Gated/Research.
*   **Hugging Face Source:** `sailor/vietnamese_rag` (contains `rag_viQuAD.json`).
*   **Strategic Stance:** Excluded from the Sailor2 fine-tuning pipeline to prevent training data contamination. ViNewsQA is discarded due to its EULA gates and focus on a health/news domain which diverges from our Wikipedia-grounded graph schema.
