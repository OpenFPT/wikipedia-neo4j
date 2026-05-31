# Thesis Defense Script

**Duration:** 15-20 minutes (target 17 min)
**Pace:** ~50 seconds per slide average, adjust per complexity

---

## Slide 1: Title (30s)

**Say:**
Good morning, committee members. My name is [Name], student ID [ID]. Today I present my thesis: Vietnamese GraphRAG — Knowledge Graph Question Answering over Wikipedia with a Local Small Language Model and Neo4j. My advisor is [Advisor Name].

**Transition:** Let me start with the presentation outline.

---

## Slide 2: Agenda (20s)

**Say:**
The presentation covers 5 main parts: first the problem and objectives, then our technical approach across 6 slides, followed by dataset construction, evaluation, and conclusions. I'll keep each section focused so we have time for discussion.

**Transition:** Let's begin with the problem we're solving.

---

## Slide 3: Problem & Research Gap (60s)

**Key message:** Three gaps motivate this work — no one has solved all three together for Vietnamese.

**Say:**
Vietnamese QA systems today have three critical limitations. First, they only handle single-hop extractive questions — you ask about one fact in one passage. No system does multi-hop reasoning over a knowledge graph in Vietnamese.

Second, hallucination. Liu et al. showed 57% of LLM-generated citations are unfaithful. Without passage-level grounding, users can't trust the answers.

Third, API dependency. Using GPT-4 or Claude means sending Vietnamese data to external servers — raising cost, latency, and data sovereignty issues for organizations that need on-premise deployment.

The bottom line: no fully-local Vietnamese KGQA system exists today.

**Transition:** These gaps define our four research objectives.

---

## Slide 4: Research Objectives (40s)

**Key message:** Four objectives, each with a concrete deliverable and measurable metric.

**Say:**
We set four objectives. O1: build a fully local GraphRAG system that runs on consumer hardware — 8GB VRAM, under 3 seconds latency. O2: produce an open multi-hop dataset, ViWiki-MHR, with 36K examples across 6 reasoning types. O3: ensure every answer is citation-grounded with deterministic validation. O4: evaluate on standard benchmarks — ViQuAD 2.0 for comparability, ViWiki-MHR for multi-hop retrieval.

The key constraint: everything runs locally, no external APIs.

**Transition:** Before our approach, let me position this against related work.

---

## Slide 5: Related Work & Positioning (60s)

**Key message:** Each existing system leaves a gap; we fill the intersection.

**Say:**
Microsoft's GraphRAG uses community summaries — powerful but enterprise-scale, not suitable for local deployment. StepChain does question decomposition over knowledge graphs but is English-only with no Vietnamese NER support. HisGraphRAG is the closest Vietnamese work, but it's domain-specific to history textbooks.

CyVerACT provides deterministic Cypher validation — we adopt this idea — but has no agent loop and no Vietnamese support. GFM-RAG requires pre-training a graph foundation model. ReflectiveRAG adds sufficiency checks but operates on flat text, no knowledge graph.

Our position fills the intersection: local SLM, typed Vietnamese KG, deterministic tool-calling, and an open dataset.

**Transition:** This leads to our three key contributions.

---continue

## Slide 6: Key Contributions (40s)

**Key message:** Three contributions — system, dataset, and SLM pipeline.

**Say:**
Contribution one: a complete GraphRAG system with typed Neo4j knowledge graph, three pluggable NER backends, hybrid retrieval with cross-encoder reranking, and a ReAct agent with four deterministic tools.

Contribution two: ViWiki-MHR — approximately 36K multi-hop QA pairs with 6 reasoning types, broken-link adversarial unanswerables, and 3-stage quality control.

Contribution three: a local SLM pipeline — Qwen2.5-7B at 4-bit quantization with Text2Cypher generation and CyVer validation, all running on 8GB VRAM.

**Transition:** Let me walk through the system architecture.

---

## Slide 7: System Architecture (50s)

**Key message:** Linear pipeline, fully local, three retrieval backends feed the agent.

**Say:**
The architecture is a linear pipeline. A user query enters the SLM orchestrator — Qwen2.5-7B-Instruct at 4-bit NF4, consuming about 5.5GB VRAM. The orchestrator drives a ReAct loop capped at 6 iterations.

The agent has access to three retrieval backends: Neo4j for structured Cypher traversal, BM25 plus dense vectors for text search, and a cross-encoder reranker — BAAI's bge-reranker-v2-m3 — for scoring.

After the agent gathers sufficient evidence, a citation verifier ensures every claim maps to a passage ID before producing the final answer.

Everything runs on a single machine — 16GB RAM, one RTX 3060 or 4060.

**Transition:** The knowledge graph is the foundation. Let me show its structure.

---

## Slide 8: Knowledge Graph Construction (50s)

**Key message:** Typed schema with pluggable NER — swap backends without changing the graph.

**Say:**
Our graph schema has three layers. Pages link to chunks via HAS_CHUNK. Chunks connect to typed entities — Person, Organization, Location, Work — through typed mention edges. Pages also link to each other via LINKS_TO for inter-article traversal.

Currently we have 998 Wikipedia pages, 19K chunks, and 3.3K extracted entities.

The NER is pluggable — three backends. Simple uses regex heuristics. Underthesea provides BIO tagging for offline Vietnamese NER. PhoNLP combines VnCoreNLP word segmentation with neural NER. All backends output the same name-type tuples that map to typed graph labels.

**Transition:** Once we have the graph, we need to translate questions into Cypher queries.

---

## Slide 9: Text2Cypher Pipeline (60s)

**Key message:** Four stages with a retry loop — inspired by CyVerACT but with an agent.

**Say:**
Text2Cypher has four stages. First, schema linking — BM25 and TF-IDF prune the full graph schema down to relevant labels and relations for this specific question.

Second, the SLM generates a Cypher query conditioned on the pruned schema.

Third, CyVer validation — deterministic checks for syntax correctness, schema alignment, and security. We block any write keywords.

Fourth, if validation fails, the error message feeds back as an observation and the SLM retries. This error-driven refinement loop is inspired by CyVerACT.

Here's a concrete example: a 2-hop Vietnamese question about the birth year of the author of "Tat den" translates to a MATCH pattern traversing from the work node back to the author node and returning the birth year.

**Transition:** Cypher is one retrieval path. Let me show the full hybrid approach.

---

## Slide 10: Hybrid Retrieval & Reranking (50s)

**Key message:** Graph-first with text fallback — reranking adds 15% accuracy.

**Say:**
The retrieval pipeline has five steps. Schema linking prunes the graph. The SLM generates read-only Cypher. Deterministic validation checks it. If the Cypher is invalid — which happens — we fall back to BM25 plus dense vector retrieval over the text chunks. Finally, a cross-encoder reranks the top-k results.

Why hybrid? Graph-only retrieval degrades when the knowledge graph is sparse — not every fact has a typed edge. Text fallback ensures coverage. And cross-encoder reranking adds approximately 15% accuracy improvement based on literature benchmarks.

**Transition:** These retrieval results feed into the ReAct agent.

---

## Slide 11: ReAct Agent (50s)

**Key message:** Constrained agent — 4 tools, 6 iterations max, sufficiency gating.

**Say:**
The agent has exactly four tools. kg_schema returns the cached graph schema. kg_query executes Cypher and returns rows or error messages. text_search does hybrid BM25 plus dense retrieval. get_passage fetches raw paragraph text for citation.

The loop runs at most 6 iterations. Each iteration: the SLM produces a thought, we parse the tool call, execute it, and check if the evidence is sufficient. If the knowledge graph returns empty results, the agent routes to text_search or abstains entirely.

Every observation is linked to a passage ID for citation tracking. This design aligns with ReflectiveRAG's sufficiency checks and C2RAG's adaptive re-retrieval.

**Transition:** Now let me cover the model and fine-tuning strategy.

---

## Slide 12: Local SLM & Fine-tuning (40s)

**Key message:** Runs now at 4-bit; QLoRA planned for Text2Cypher specialization.

**Say:**
The base model is Qwen2.5-7B-Instruct, loaded lazily with 4-bit NF4 quantization — about 5.5GB VRAM. It exposes both a generate and chat interface.

For fine-tuning, we plan QLoRA — NF4 with double quantization, LoRA rank 32 across all linear layers, trained on 10 to 15K Text2Cypher pairs. Then DPO alignment for JSON tool-call compliance. The target is over 95% executable Cypher output.

This is planned work — the base model already functions for the agent loop, fine-tuning will improve Cypher generation accuracy.

**Transition:** Let me now present our dataset.

---

## Slide 13: Data Sources (50s)

**Key message:** Two complementary datasets — external benchmark + our multi-hop contribution.

**Say:**
We use two datasets. UIT-ViQuAD 2.0 is the external benchmark — 39.5K QA pairs from 174 Wikipedia articles, peer-reviewed, with 12K unanswerable questions in SQuAD 2.0 style. This gives us comparability with published results.

ViWiki-MHR is our contribution — approximately 36K multi-hop QA pairs with 6 reasoning types, broken-link adversarial unanswerables, and gold passage IDs with executable Cypher queries. The underlying Vietnamese Wikipedia corpus is now pinned and published as `Keithsel/viwiki-20260523` on Hugging Face, produced from the raw 2026-05-23 MediaWiki XML dump.

The breakdown: 23K single-hop from ViQuAD, 5K unanswerable, 7K multi-hop from KG walks with LLM rewrite, and about 1K adversarial multi-hop using broken links. This last category is the first Vietnamese benchmark for multi-hop unanswerability.

**Transition:** Let me explain how we generate the multi-hop portion.

---

## Slide 14: Dataset Generation Pipeline (50s)

**Key message:** KG walks to templates to optional LLM rewrite to 3-stage QC.

**Say:**
The generation pipeline starts with corpus preparation: raw Vietnamese Wikipedia XML is streamed, cleaned into article text, and exported to cleaned and raw Parquet shards. This processed corpus is published as `Keithsel/viwiki-20260523`. From the ingested graph, we then extract KG walks — 2-hop, 3-hop, and broken-link paths. Next, we apply Vietnamese question templates per entity type and hop count. Third, optionally, an LLM rewrites the template question for naturalness. Fourth, 3-stage quality control: well-formedness check, grounding match against the graph, and deduplication.

We cover six reasoning types: lookup, bridge, comparison, intersection, temporal, and fan-out.

The adversarial unanswerables are inspired by BRINK — we break one link in a valid KG path, creating questions that look coherent but have no valid answer in the graph.

**Transition:** We also integrate ViQuAD 2.0 as an external evaluation benchmark.

---

## Slide 15: ViQuAD 2.0 Integration (40s)

**Key message:** External benchmark for objectivity — measures the generative-extractive gap.

**Say:**
We integrate ViQuAD 2.0 through a four-step pipeline: load from HuggingFace, deduplicate by title and text hash, ingest new passages into the knowledge graph, and run end-to-end evaluation on the validation split.

Why ViQuAD 2.0? Objectivity — it's external, not self-generated. Comparability — CafeBERT achieves 77.5% exact match, giving us a reference point. Abstain testing — 12K impossible questions test our system's ability to say "I don't know." And it measures the generative gap — our system produces free-form answers, not extractive spans, so Token F1 is the fair metric.

Success criteria: Token F1 above 0.40, context hit rate above 0.60, abstain accuracy above 0.60.

**Transition:** Now the evaluation framework and results.

---

## Slide 16: Evaluation Framework & Results (40s)

**Key message:** Multi-metric evaluation across retrieval, answer quality, robustness, and efficiency.

**Say:**
We evaluate across five categories. Retrieval quality: hit rate and MRR. Answer quality: token F1 and exact match. Robustness: abstain accuracy on impossible questions. Efficiency: average latency. Groundedness: citation faithfulness.

We compare against three baselines: vector-only dense retrieval, BM25-only sparse retrieval, and graph-only Cypher traversal. Our hybrid approach combines all three with reranking.

[If results available: present numbers. If placeholder: "Full evaluation is in progress, here are our targets based on literature."]

**Transition:** Let me show a quick demo of the system in action.

---

## Slide 17: Live Demo (40s)

**Key message:** Show the multi-hop reasoning trace end-to-end.

**Say:**
Here's a sample 2-hop query: "Who founded the organization headquartered in Hanoi?" The reasoning requires traversing from Location to Organization to Person.

The agent trace shows: it calls kg_query with a MATCH pattern, gets back a result with the person's name and the passage ID, and returns the answer with citation. Every step is traceable — the committee can verify the answer by checking chunk-0042 in the graph.

[If live demo: switch to terminal and run the query.]

**Transition:** Let me summarize and discuss future directions.

---

## Slide 18: References (skip)

**Note:** Don't present this slide. Flip to it only if committee asks about a specific citation.

---

## Slide 19: Conclusion & Future Work (60s)

**Key message:** Working system today, clear roadmap for improvement.

**Say:**
To summarize what we've achieved: a complete end-to-end local GraphRAG system, a typed Vietnamese knowledge graph with 998 pages and 19K chunks, a ReAct agent with citation tracking, a dataset generation pipeline integrated with ViQuAD 2.0, and a multi-metric evaluation framework.

For future work: QLoRA fine-tuning for Text2Cypher will improve Cypher generation accuracy. DPO alignment will ensure tool-call compliance. Community detection using Louvain will enable global search capabilities. Query decomposition following StepChain will handle more complex multi-hop questions. And we plan to scale to 5000+ pages.

Thank you for your attention. I'm happy to take questions.

---

## Anticipated Questions & Backup Slides

### Q: "How does this compare to Naive RAG?"
**Flip to:** Backup slide 1
**Key points:** Naive RAG uses flat chunks, poor at multi-hop. GraphRAG uses communities but is expensive. Ours uses typed KG + text hybrid — small context windows via Cypher, medium cost on consumer hardware.

### Q: "How do broken-link adversarial questions work?"
**Flip to:** Backup slide 2
**Key points:** Take a valid 3-hop path, fabricate the last entity. Question looks coherent but the final link doesn't exist. Forces agent to verify each step. First Vietnamese benchmark for this.

### Q: "What are the QLoRA training details?"
**Flip to:** Backup slide 3
**Key points:** 4-bit NF4, double quantization, LoRA rank 32, alpha 64, all linear layers, 10-15K pairs, Paged AdamW.

### Q: "Why Qwen2.5 and not PhoBERT or VinaLLaMA?"
**Answer:** PhoBERT is encoder-only — can't generate Cypher or free-form answers. VinaLLaMA is 13B — too large for 8GB VRAM at acceptable quantization. Qwen2.5-7B at 4-bit fits in 5.5GB, supports Vietnamese well, and has strong instruction-following.

### Q: "What about latency? 3 seconds seems slow."
**Answer:** 3s is the target end-to-end including up to 6 agent iterations. Each Cypher execution is ~50ms. The bottleneck is SLM inference — 7B model at 4-bit generates ~30 tokens/second. For single-hop questions needing 1-2 iterations, latency is under 1.5s.

### Q: "Why not use GPT-4 for better accuracy?"
**Answer:** Data sovereignty — Vietnamese organizations may not want to send data externally. Cost — GPT-4 at scale is expensive. Reproducibility — local model means deterministic results. Our system is designed for on-premise deployment.

### Q: "How do you handle entity disambiguation?"
**Answer:** Currently exact string matching with type constraints. The typed schema helps — "Ha Noi" as Location vs Organization are different nodes. Future work: entity linking with embedding similarity for fuzzy matching.

### Q: "998 pages seems small. What about coverage?"
**Answer:** 998 pages is prototype scale for validation. The pipeline scales — async batch processing. We plan 5000+ pages. Even at 998 pages, 19K chunks and 3.3K entities demonstrate multi-hop reasoning across diverse topics.

---

## Timing Checklist

| Section | Slides | Target | Cumulative |
|---------|--------|--------|------------|
| Opening + Agenda | 1-2 | 0:50 | 0:50 |
| Problem + Objectives | 3-4 | 1:40 | 2:30 |
| Related Work + Contributions | 5-6 | 1:40 | 4:10 |
| Architecture + KG + Text2Cypher | 7-9 | 2:40 | 6:50 |
| Retrieval + Agent + SLM | 10-12 | 2:20 | 9:10 |
| Dataset + Generation + ViQuAD2 | 13-15 | 2:20 | 11:30 |
| Evaluation + Demo | 16-17 | 1:20 | 12:50 |
| Conclusion | 19 | 1:00 | 13:50 |
| Buffer | — | 1:10 | 15:00 |

**Total:** ~15 min presentation + 5 min Q&A = 20 minutes

---

## Presentation Tips

- Speak slowly on slides 3, 9, 14 (most technical)
- Point to diagram elements when explaining architecture (slide 7)
- Pause after "first Vietnamese benchmark" claim — let it land
- On slide 16, if results are placeholder, be honest about it
- Practice the demo trace walkthrough so it feels natural
- Keep eye contact during Q&A, don't read backup slides verbatim
- Drink water before starting — 15 minutes of continuous speaking
