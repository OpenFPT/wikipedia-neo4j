# Related Work

## Vietnamese Multi-Hop QA and Knowledge Graph QA
Vietnamese multi-hop QA remains a low-resource area. **ViMQA** provides a human-annotated multi-hop benchmark with supporting-fact supervision, but its distribution is gated and restricts downstream reuse, limiting reproducible research. Domain-specific KG efforts such as **VietMedKG** and **ViHERMES** demonstrate the value of typed relations and graph traversal for multi-hop reasoning, especially in regulated domains. These efforts motivate the need for an open, scalable Vietnamese KGQA system that can be trained and evaluated without data access barriers.

## Retrieval-Augmented Generation (RAG) and GraphRAG
RAG surveys (Gao et al.; Fan et al.) describe an evolution from Naive to Advanced and Modular RAG, emphasizing retrieval quality, groundedness, and answer relevance. **GraphRAG** (Edge et al.) demonstrates dual search—local entity neighborhood retrieval and global community summarization—but its enterprise-scale global search is costly for local setups. Our design adopts **local graph search** and replaces global summarization with deterministic Cypher traversal plus hybrid text fallback, consistent with findings that graph-only pipelines degrade under KG sparsity.

## Agentic RAG and Tool-Driven Orchestration
Agentic RAG surveys and SoK analyses highlight the role of planning, reflection, and tool use but also identify risks: hallucination propagation, unbounded loops, and coordination failures. **ReflectiveRAG** and **C2RAG** show that sufficiency checks and adaptive re-retrieval improve robustness under noisy or incomplete KGs. We align with these insights by implementing a **capped ReAct loop** with deterministic tool calls and explicit sufficiency gating, avoiding multi-agent coordination overhead.

## Text2Cypher and Deterministic Verification
Text-to-Cypher translation is a core enabler for KGQA. **Ozsoy et al.** show that schema-conditioned prompts and multilingual fine-tuning dramatically improve Cypher generation with small models. **CyVerACT** introduces a deterministic validation workflow that feeds structured compiler errors back to the generator, improving execution rates. Our approach inherits this pattern and integrates **deterministic syntax-schema validation** directly into a ReAct loop, replacing stochastic evaluators.

## Local SLM Fine-Tuning and Alignment
QLoRA (Dettmers et al.) enables parameter-efficient training on consumer hardware, while DPO (Rafailov et al.) provides stable preference-based alignment without RL. Empirical studies (TinyLLM; Birkholm et al.) show that targeted SFT + DPO on tool-calling tasks can outperform much larger general models. This supports our focus on **local 7–8B models**, fine-tuned for Vietnamese Text2Cypher and strict tool formatting.

## Hybrid Retrieval and Vietnamese Baselines
Vietnamese baselines such as **ViWiQA** validate hybrid sparse+dense retrieval, and **HisGraphRAG** shows entity-grounded graph context improves precision. **HybridRAG** confirms that graph+text hybrid pipelines outperform graph-only systems on multi-hop queries. We adopt a **WRRF (Weighted Reciprocal Rank Fusion)** approach combining BM25, vector similarity, graph traversal, and community-based retrieval, with cross-encoder reranking for final scoring.

# Method

## Overview
We propose a fully local Vietnamese KGQA system that combines a typed Neo4j knowledge graph with a tool-constrained SLM orchestrator (AITeamVN/Vi-Qwen2-7B-RAG). The system answers multi-hop questions by translating natural language into Cypher, executing deterministic graph queries, and validating groundedness with explicit citations. The architecture runs on consumer hardware (16GB RAM, 8GB VRAM) without external APIs.

## Knowledge Graph Construction
We build a typed KG from a pinned Vietnamese Wikipedia snapshot published as `Keithsel/viwiki-20260523` on Hugging Face. The snapshot is produced from the raw 2026-05-23 MediaWiki XML dump using `scripts/viwiki_processing/`: XML is streamed with `lxml`, main-namespace articles are retained, wikitext is cleaned with `mwparserfromhell`, and both cleaned text and raw wikitext are exported as Parquet shards. Nodes include typed entities (e.g., `Person`, `Place`, `Organization`, `Event`, `TácPhẩm`) with temporal/numeric attributes; relations are typed and directed (e.g., `:SÁNG_TÁC`, `:GIA_NHẬP`, `:TRỤ_SỞ`). Each entity is linked to passage nodes containing the source paragraph, enabling passage-level citation tracing. **Entity resolution** merges diacritic variants and known Vietnamese aliases (e.g., 'Bác Hồ' → 'Hồ Chí Minh'). **Relation extraction** uses LLM-based typed extraction with six relation types (FOUNDED_BY, LOCATED_IN, BORN_IN, MEMBER_OF, PART_OF, CREATED_BY). **Community detection** (Louvain) groups related entities and generates pre-computed summaries for global-context retrieval.

## Tool-Constrained Orchestrator
We restrict the orchestrator to six deterministic tools:
1. **`kg_schema()`**: returns cached schema (labels, relations, properties).
2. **`kg_query(cypher)`**: executes Cypher and returns rows or compiler errors.
3. **`text_search(query, k)`**: hybrid BM25 + dense retrieval.
4. **`get_passage(passage_id)`**: fetches raw text for verification.
5. **`entity_neighborhood(entity, hops)`**: returns typed neighbors within k hops.
6. **`path_search(entity_a, entity_b, max_hops)`**: finds shortest paths between entities.

The orchestrator executes a **ReAct loop** capped at 6 iterations. The Thought step performs sufficiency checking; if graph evidence is empty or inconsistent, the system switches to text retrieval or abstains.

For complex multi-hop questions, the system applies **question decomposition** to break the query into sub-questions, executes **multiple trajectories** with temperature scaling, and uses **majority voting** to select the most consistent answer. Complexity is detected automatically based on question structure.

## Text2Cypher Translation Pipeline
We implement a four-stage pipeline:
1. **Schema linking**: prune schema using BM25/TF-IDF over labels/relations.
2. **SLM generation**: generate Cypher from Vietnamese query + pruned schema.
3. **Deterministic validation**: syntax, schema, and safety checks; destructive clauses are stripped.
4. **Error refinement**: compiler errors are fed back to the model as observations within the ReAct loop.

This replaces vector-only retrieval with executable graph traversal, enabling precise multi-hop reasoning.

## Dataset: ViWiki-MHR
We introduce **ViWiki-MHR** (~36K), an open, multi-hop Vietnamese QA dataset grounded to `gold_passage_ids`. It is generated on top of the reproducible `Keithsel/viwiki-20260523` corpus and combines:
- UIT-ViQuAD 2.0 single-hop answerable/unanswerable subsets.
- Synthetic multi-hop answerable and adversarial unanswerable questions generated via KG walks.

Each sample includes `reasoning_type`, `num_hops`, `cypher_query`, and `decomposition_annotations`.
Quality control uses a 3-stage pipeline: grounding match, NLI entailment, LLM coherence check, plus human spot-checks.

## Fine-Tuning and Alignment
We fine-tune a local SLM (AITeamVN/Vi-Qwen2-7B-RAG) using:
- **QLoRA** (NF4, double quantization, LoRA on all linear layers) for Vietnamese Text2Cypher.
- **DPO** alignment to enforce JSON tool-call formatting and schema compliance.

The result is a lightweight model capable of reliable tool calling on consumer hardware.

## Evaluation
We evaluate using RAG Triad metrics (context relevance, faithfulness, answer relevance), with explicit passage-ID verification. We benchmark on both ViWiki-MHR and UIT-ViQuAD2.0 (achieving 72.6% context hit rate on the latter). We include hallucination taxonomy from ViHallu (intrinsic/extrinsic) and adversarial missing-link tests inspired by BRINK. Baselines include vector-only and BM25-only retrieval, graph-only traversal, and hybrid retrieval.
