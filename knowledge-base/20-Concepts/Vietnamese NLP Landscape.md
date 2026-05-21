# Vietnamese NLP Landscape

Building QA systems for Vietnamese introduces unique linguistic and resource challenges that are absent in English-only environments. Successful implementation requires adapting tokenization, entity alignment, and search indices to accommodate these properties.

---

## The Resource & Modeling Gap

Vietnamese is historically categorized as a medium-to-low-resource language in the NLP literature. While extensive corpora exist for basic tasks, advanced multi-hop reasoning datasets are extremely scarce, making previous datasets like ViMQA critical baselines and inspiring the creation of our own open-license, retrieval-augmented **ViWiki-MHR** corpus (30K examples).

### Core Pre-trained Vietnamese Models
To construct a hybrid retrieval pipeline, we leverage specialized Vietnamese transformer models or Southeast Asian open frontier models:
*   **PhoBERT (VinAI):** A state-of-the-art monolingual RoBERTa-based model trained on a large Vietnamese dataset. It is highly effective for Named Entity Recognition (NER), word segmentation, and semantic embeddings.
*   **Vistral-7B / SeaLLM-7B:** Modern open LLMs explicitly pre-trained or instruction-tuned for Southeast Asian languages, representing competent base models for our local deployment.
*   **Qwen 2.5-7B-Instruct:** A highly performant multilingual base model exhibiting strong Vietnamese syntax capability, excellent tokenization efficiency, and solid native tool-calling features.

---

## Key Linguistic Challenges

Any Vietnamese QA architecture (especially GraphRAG and KG-QA) must solve three core linguistic issues:

### 1. Tokenization & Word Segmentation
In English, words are separated by spaces. In Vietnamese, spaces separate **syllables**, and a single word can consist of one, two, or more syllables:
*   *Example:* "học sinh" (student - 2 syllables, 1 word), "bóng đá" (football - 2 syllables, 1 word).
*   **Impact:** Standard whitespace tokenizers treat these as separate tokens, which corrupts word embedding representations.
*   **Solution:** We must use a specialized Vietnamese word segmenter like `underthesea` or `pyvi` during pre-processing to bind syllables together (e.g., "học sinh" -> "học_sinh") before computing vector embeddings or running NER.

### 2. Diacritics & Accents
Vietnamese uses a complex diacritic system to mark tones and distinct vowel sounds:
*   *Example:* Ho (surname / dry), Hồ (lake / surname), Hổ (tiger), Hộ (household).
*   **Impact:** A user typing without accents (e.g., "John O'Shea gia nhap clb nao") or slightly misspelling a diacritic will cause direct text matches to fail completely.
*   **Solution:** Our BM25 index and vector retrieval tools must employ accent-insensitive analyzers or fuzzy matching scripts during user query parsing.

### 3. Conversational Pronoun Dropping (Pro-drop)
Vietnamese is a pro-drop language where subjects or objects are frequently omitted in subsequent sentences if they can be inferred from context:
*   *Example Question:* "Được thành lập ở đâu?" (Where was [it] established?).
*   **Impact:** Semantic search over raw text blocks fails because the keyword (the entity being asked about) is missing from the query.
*   **Solution:** We will implement the entity resolution and clarification pipeline inspired by URASys (2025) to identify missing nodes and reconstruct clean search queries.

---

## The GraphRAG Opportunity

Traditional RAG suffers from semantic fragmentation in Vietnamese due to these linguistic complexities. By shifting to a **Knowledge Graph-based RAG**, we bypass the limitations of raw text chunking:
1.  **Semantic Triples:** Relationships are explicitly defined (e.g., `(Sông_Đà)-[:PHỤ_LƯU_LỚN_NHẤT]->(Sông_Hồng)`). Word segmentation errors do not affect graph traversal.
2.  **Entity Resolution:** All spelling and diacritic variations (e.g., Hồ Chí Minh, Bác Hồ, Nguyễn Tất Thành) can be merged into a single database node via Entity Alignment, ensuring that queries searching for any variant retrieve the exact same consolidated knowledge sub-graph.
