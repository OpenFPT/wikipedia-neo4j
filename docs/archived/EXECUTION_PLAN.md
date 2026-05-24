# ViWiki-MHR — Execution Plan (3 tháng)

> Timeline: 12 tuần | 4 thành viên | Mỗi người đóng góp mọi phase

---

## Team Roles

| Member | Strengths | Primary Ownership |
|--------|-----------|-------------------|
| **TrungHQ (T)** | ML/NLP, model training, system architecture | SLM stack, fine-tuning, ReAct agent core |
| **KhoiPND (K)** | Backend engineering, data pipelines | Neo4j ingestion, KG construction, API integration |
| **AnhNQ (A)** | Data engineering, scripting, infrastructure | Dataset generation, QC pipeline, evaluation |
| **ThienTD (D)** | Testing, documentation, verification | Test suites, human QC, docs, demo |

---

## Phase 1: Foundation Setup (Tuần 1–2)

**Goal:** Hạ tầng sẵn sàng, SLM chạy local, NER pipeline hoạt động, Wikipedia dump loaded.

| # | Task | Owner | Deliverable |
|---|------|-------|-------------|
| 1.1 | Setup dev environment: Docker Compose (Neo4j + Qdrant), `.env`, Makefile targets | K | `docker-compose.yml`, `Makefile` updated |
| 1.2 | Download Vietnamese Wikipedia dump (HF `wikimedia/wikipedia` 20231101.vi), convert to paragraphs | K | `scripts/download_viwiki.py` → `data/viwiki_paragraphs.parquet` |
| 1.3 | Setup vLLM/llama.cpp server cho Qwen2.5-7B-Instruct 4-bit (GPTQ hoặc AWQ) | T | `scripts/run_slm.sh`, verify Vietnamese generation |
| 1.4 | Bake-off: benchmark 3 candidates (Qwen2.5-7B, SeaLLM-7B, Sailor2-8B) trên 50 câu hỏi | T | `reports/slm_bakeoff.md` với latency, quality scores |
| 1.5 | Implement Vietnamese word segmentation module (underthesea + VnCoreNLP fallback) | A | `src/vietnamese_nlp.py` |
| 1.6 | Implement Vietnamese NER pipeline: extract Person, Organization, Location, Work | A | `src/ner_pipeline.py` |
| 1.7 | Design & implement extended Neo4j schema (typed nodes + relationships + constraints) | K | `scripts/setup_neo4j_schema.py` |
| 1.8 | Viết unit tests cho word segmentation (edge cases: tên riêng, viết tắt, số) | D | `tests/test_vietnamese_nlp.py` |
| 1.9 | Viết unit tests cho NER pipeline (precision/recall trên 20 sample paragraphs) | D | `tests/test_ner_pipeline.py` |
| 1.10 | Viết test cho Neo4j schema (verify indexes, constraints, node labels exist) | D | `tests/test_schema_setup.py` |
| 1.11 | Document architecture decisions: model choice rationale, schema design | D | `docs/architecture_decisions.md` |

**Exit Criteria:**
- `make check` passes
- SLM generates coherent Vietnamese text locally
- NER extracts entities from 10 Wikipedia articles correctly
- Neo4j schema deployed with all typed nodes/relationships

---

## Phase 2: KG Construction & Agent Baseline (Tuần 3–5)

**Goal:** Knowledge Graph populated, 4 agent tools functional, ReAct loop answers 100 questions.

| # | Task | Owner | Deliverable |
|---|------|-------|-------------|
| 2.1 | Build full ingestion pipeline: paragraphs → segment → NER → resolve → write Neo4j | K | Refactored `src/ingest.py` |
| 2.2 | Implement entity resolution: merge diacritic variants, aliases (Hồ Chí Minh = Bác Hồ = Nguyễn Tất Thành) | K | `src/entity_resolution.py` |
| 2.3 | Build relation extraction: rule-based patterns + LLM-assisted for complex rels | T | `src/relation_extraction.py` |
| 2.4 | Run full ingestion on 220K paragraphs, monitor & fix failures | K | Neo4j populated, ingestion report |
| 2.5 | Setup Qdrant local vector store + BM25 sparse index over paragraphs | A | `src/vector_store.py` |
| 2.6 | Generate embeddings cho toàn bộ paragraphs (bge-m3 hoặc multilingual-e5) | A | Qdrant collection populated |
| 2.7 | Implement 4 agent tools: `kg_schema()`, `kg_query()`, `text_search()`, `get_passage()` | K | `src/agent_tools.py` |
| 2.8 | Implement ReAct agent loop (Thought→Action→Observation, 6-iter cap, JSON actions) | T | `src/react_agent.py` |
| 2.9 | Implement citation tracking & groundedness post-processor | T | `src/groundedness.py` |
| 2.10 | Build `MODEL_MODE` toggle (`local` vs `api`) trong config | A | Updated `src/config.py` |
| 2.11 | Curate 100 câu hỏi test (mix 1-hop, 2-hop, comparison) cho baseline eval | D | `data/eval/baseline_100q.jsonl` |
| 2.12 | Viết integration test: ingest 5 articles → ask 10 questions → verify answers | D | `tests/test_integration_e2e.py` |
| 2.13 | Viết unit tests cho ReAct agent (mock tools, verify cap, format compliance) | D | `tests/test_react_agent.py` |
| 2.14 | Viết unit tests cho groundedness verifier | D | `tests/test_groundedness.py` |
| 2.15 | Run baseline evaluation trên 100 câu hỏi, analyze failure modes | T | `reports/baseline_100q_results.md` |

**Exit Criteria:**
- Neo4j contains 50K+ entities, 100K+ relationships
- ReAct agent answers 100 questions end-to-end
- Groundedness check flags ungrounded answers correctly
- All tests pass

---

## Phase 3: ViWiki-MHR Dataset Generation (Tuần 6–8)

**Goal:** Dataset 36K mẫu hoàn chỉnh, 3-stage QC pass, 1500+ mẫu human-verified.

| # | Task | Owner | Deliverable |
|---|------|-------|-------------|
| 3.1 | Reformat UIT-ViQuAD 2.0 → ViWiki-MHR schema (23K answerable + 5K unanswerable) | A | `data/viwiki_mhr/uit_viquad_reformatted.jsonl` |
| 3.2 | Map ViQuAD passages → `gold_passage_ids` trong Wikipedia corpus | A | Passage ID mapping complete |
| 3.3 | Build KG walk template engine: extract valid 2-hop & 3-hop paths từ Neo4j | T | `scripts/kg_walk_generator.py` |
| 3.4 | Build NL rewriter: KG path → natural Vietnamese question (frontier LLM) | T | `scripts/question_rewriter.py` |
| 3.5 | Generate ~7K multi-hop answerable QA pairs (bridge, comparison, intersection, temporal) | T | `data/viwiki_mhr/synthetic_multihop.jsonl` |
| 3.6 | Build adversarial broken-link generator (swap entity/relation at final hop) | T | `scripts/adversarial_generator.py` |
| 3.7 | Generate ~1K adversarial multi-hop unanswerable pairs | T | `data/viwiki_mhr/adversarial_unanswerable.jsonl` |
| 3.8 | Generate gold Cypher queries cho mỗi multi-hop sample | K | `cypher_query` field populated |
| 3.9 | Generate decomposition annotations (sub-questions + sub-answers) | K | `decomposition_annotations` field populated |
| 3.10 | Implement QC Stage 1: Grounding filter (answer ∈ gold passages) | A | `scripts/qc_grounding.py` |
| 3.11 | Implement QC Stage 2: NLI entailment filter (cross-encoder model) | A | `scripts/qc_nli.py` |
| 3.12 | Implement QC Stage 3: Well-formedness filter (LLM verifier) | A | `scripts/qc_wellformed.py` |
| 3.13 | Human verification: annotate 1500+ test split samples (answerable check, quality) | D | `data/viwiki_mhr/human_verified.jsonl` |
| 3.14 | Build annotation tool/spreadsheet cho human verification workflow | D | Annotation guide + tool |
| 3.15 | Compile final dataset: merge all sources, apply QC, generate train/dev/test splits | A | `data/viwiki_mhr/final/{train,dev,test}.jsonl` |
| 3.16 | Viết dataset statistics report (distribution by type, hops, answerable ratio) | D | `reports/dataset_statistics.md` |
| 3.17 | Prepare HuggingFace dataset card + upload automation | K | `scripts/upload_hf_dataset.py`, README card |

**Exit Criteria:**
- 36K total samples (28K single-hop + 7K multi-hop + 1K adversarial)
- All 3 QC stages pass rate >95%
- 1500+ human-verified test samples with 100% ground truth accuracy
- Dataset card ready for HF upload

---

## Phase 4: QLoRA Fine-tuning (Tuần 9–10)

**Goal:** Text2Cypher adapter + ReAct adapter trained, measurable improvement over base.

| # | Task | Owner | Deliverable |
|---|------|-------|-------------|
| 4.1 | Build Text2Cypher training corpus từ KG walks: (question, schema, gold Cypher) ~12K pairs | T | `data/finetune/text2cypher_train.jsonl` |
| 4.2 | Build ReAct format corpus: (question, tool sequence, observations, final answer) ~3K traces | T | `data/finetune/react_format_train.jsonl` |
| 4.3 | Implement QLoRA training script (BitsAndBytes NF4, LoRA r=32, α=64, paged_adamw) | T | `scripts/train_qlora.py` |
| 4.4 | Train Text2Cypher adapter (Stage 1: instruction tuning) | T | `models/text2cypher_adapter/` |
| 4.5 | Build DPO preference pairs: chosen (valid Cypher) vs rejected (hallucinated/fluff) | K | `data/finetune/dpo_pairs.jsonl` |
| 4.6 | Train DPO alignment (Stage 2) | T | `models/text2cypher_dpo/` |
| 4.7 | Train ReAct format adapter | T | `models/react_adapter/` |
| 4.8 | Implement adapter evaluation: Cypher compilation rate, schema accuracy, format compliance | A | `scripts/eval_adapter.py` |
| 4.9 | Benchmark adapter vs base SLM trên 500 held-out Text2Cypher pairs | A | `reports/adapter_benchmark.md` |
| 4.10 | Viết tests cho training data format validation (schema, field completeness) | D | `tests/test_finetune_data.py` |
| 4.11 | Viết tests cho adapter inference (load model, run 30 queries, verify output format) | D | `tests/test_adapter_inference.py` |
| 4.12 | Setup model versioning & checkpoint management | K | `models/README.md`, checkpoint naming convention |

**Exit Criteria:**
- Text2Cypher compilation rate >90% (vs base ~60%)
- DPO eliminates conversational fluff (0% preamble in output)
- ReAct adapter produces valid JSON actions 95%+ of the time
- Measurable EM/F1 improvement on held-out set

---

## Phase 5: System Evaluation & Benchmarking (Tuần 11)

**Goal:** Full evaluation tables, ablation study, hallucination measurement complete.

| # | Task | Owner | Deliverable |
|---|------|-------|-------------|
| 5.1 | Implement evaluation harness: EM, F1, retrieval recall@k, Cypher compile rate | A | `src/evaluation.py` |
| 5.2 | Run Baseline 1: Base SLM zero-shot prompting (no RAG, no fine-tune) | T | `reports/eval/baseline_prompting.json` |
| 5.3 | Run Baseline 2: Vector-only RAG (BM25 + dense retrieval, no KG) | K | `reports/eval/vector_rag.json` |
| 5.4 | Run Baseline 3: KG-only (fine-tuned Cypher, no text fallback) | K | `reports/eval/kg_only.json` |
| 5.5 | Run Full System: fine-tuned ReAct + hybrid (KG + text) + groundedness | T | `reports/eval/full_system.json` |
| 5.6 | Ablation: with/without DPO, with/without groundedness, with/without entity resolution | T | `reports/ablation_study.md` |
| 5.7 | Hallucination test: run adversarial unanswerable subset, measure false-positive rate | T | `reports/hallucination_rate.md` |
| 5.8 | Per-reasoning-type breakdown (bridge, comparison, intersection, temporal, fan-out) | A | `reports/per_type_breakdown.md` |
| 5.9 | Compile LaTeX results tables + figures | A | `paper/tables/` |
| 5.10 | Manual QA: run 50 diverse queries, document failure modes & edge cases | D | `reports/manual_qa_testing.md` |
| 5.11 | Viết tests cho evaluation metrics (known-answer pairs, verify EM/F1 math) | D | `tests/test_evaluation.py` |
| 5.12 | Error analysis: categorize failures (retrieval miss, Cypher error, hallucination, timeout) | D | `reports/error_analysis.md` |

**Exit Criteria:**
- All 4 system configurations evaluated on full test set
- Ablation table shows contribution of each component
- Hallucination rate measured on adversarial subset
- Error taxonomy documented

---

## Phase 6: Paper, Demo & Release (Tuần 12)

**Goal:** Paper submitted, code/dataset/model published, demo ready.

| # | Task | Owner | Deliverable |
|---|------|-------|-------------|
| 6.1 | Write paper: Abstract, Introduction, Related Work | T | `paper/main.tex` sections |
| 6.2 | Write paper: System Architecture & Method | T | `paper/main.tex` sections |
| 6.3 | Write paper: Dataset Construction (ViWiki-MHR) | A | `paper/main.tex` section |
| 6.4 | Write paper: Experiments & Results | A | `paper/main.tex` section |
| 6.5 | Build Gradio web demo (query → reasoning trace → cited answer) | K | `src/app_gradio.py` |
| 6.6 | Write one-line install script + quickstart README | K | `setup.sh`, updated `README.md` |
| 6.7 | Final code cleanup: type hints, remove dead code, consistent style | K | Clean codebase |
| 6.8 | Publish dataset to HuggingFace (CC-BY-SA) | A | HF dataset repo |
| 6.9 | Publish fine-tuned adapter to HuggingFace | T | HF model repo |
| 6.10 | Create system architecture diagrams (draw.io/mermaid) | D | `paper/figures/` |
| 6.11 | Record demo video (3 phút: problem → system → live query → answer) | D | `docs/demo.mp4` |
| 6.12 | Proofread paper: grammar, citations, format compliance | D | Final `paper/main.tex` |
| 6.13 | Prepare presentation slides cho defense | D | `docs/slides.pdf` |

**Exit Criteria:**
- Paper complete & formatted
- HF dataset + model published
- Gradio demo functional
- Demo video + slides ready

---

## Timeline Gantt (12 tuần)

```
Tuần:  1   2   3   4   5   6   7   8   9  10  11  12
       ├───┤───┤───┤───┤───┤───┤───┤───┤───┤───┤───┤
Phase1 ████████
Phase2         ████████████████
Phase3                         ████████████████
Phase4                                         ████████
Phase5                                                 ████
Phase6                                                     ████

Overlap:
- Phase 3.1 (ViQuAD reformat) có thể bắt đầu song song Phase 2
- Phase 4.1 (training corpus) bắt đầu cuối Phase 3
- Phase 6 (paper writing) bắt đầu draft từ tuần 10
```

---

## Per-Member Timeline

### TrungHQ (T) — 24 tasks

```
Tuần 1-2:  SLM setup, bake-off (1.3, 1.4)
Tuần 3-5:  Relation extraction, ReAct agent, groundedness, baseline eval (2.3, 2.8, 2.9, 2.15)
Tuần 6-8:  KG walks, NL rewriter, multi-hop gen, adversarial gen (3.3-3.7)
Tuần 9-10: QLoRA training (all stages), DPO alignment (4.1-4.4, 4.6, 4.7)
Tuần 11:   Run full system eval, ablation, hallucination test (5.2, 5.5-5.7)
Tuần 12:   Write paper core sections, publish model (6.1, 6.2, 6.9)
```

### KhoiPND (K) — 16 tasks

```
Tuần 1-2:  Docker setup, Wiki dump, Neo4j schema (1.1, 1.2, 1.7)
Tuần 3-5:  Full ingestion pipeline, entity resolution, agent tools (2.1, 2.2, 2.4, 2.7)
Tuần 6-8:  Gold Cypher generation, decomposition annotations, HF upload prep (3.8, 3.9, 3.17)
Tuần 9-10: DPO pairs generation, model versioning (4.5, 4.12)
Tuần 11:   Run vector-only & KG-only baselines (5.3, 5.4)
Tuần 12:   Gradio demo, README, code cleanup (6.5-6.7)
```

### AnhNQ (A) — 18 tasks

```
Tuần 1-2:  Vietnamese NLP module, NER pipeline (1.5, 1.6)
Tuần 3-5:  Qdrant + BM25 setup, embeddings, MODEL_MODE toggle (2.5, 2.6, 2.10)
Tuần 6-8:  ViQuAD reformat, QC pipeline (3 stages), final dataset compile (3.1, 3.2, 3.10-3.12, 3.15)
Tuần 9-10: Adapter evaluation script, benchmark (4.8, 4.9)
Tuần 11:   Evaluation harness, per-type breakdown, LaTeX tables (5.1, 5.8, 5.9)
Tuần 12:   Write dataset & experiments paper sections, HF publish (6.3, 6.4, 6.8)
```

### ThienTD (D) — 17 tasks

```
Tuần 1-2:  Tests cho NLP, NER, schema; architecture docs (1.8-1.11)
Tuần 3-5:  100q curation, integration tests, ReAct tests, groundedness tests (2.11-2.14)
Tuần 6-8:  Human verification 1500+ samples, annotation tool, dataset stats (3.13, 3.14, 3.16)
Tuần 9-10: Tests cho training data format, adapter inference (4.10, 4.11)
Tuần 11:   Manual QA testing, eval tests, error analysis (5.10-5.12)
Tuần 12:   Diagrams, demo video, proofread, slides (6.10-6.13)
```

---

## Critical Dependencies

```
[1.2 Wiki dump] + [1.5 NER] + [1.7 Schema] ──► [2.1 Ingestion]
[2.1 Ingestion] ──► [2.7 Agent Tools] ──► [2.8 ReAct Agent]
[2.1 Ingestion] ──► [3.3 KG Walks] ──► [3.5 Multi-hop Gen]
[3.15 Final Dataset] ──► [4.1 Training Corpus] ──► [4.4 QLoRA Train]
[4.6 DPO Model] + [4.7 ReAct Adapter] ──► [5.5 Full System Eval]
[5.5 Full Eval] ──► [6.1 Paper Writing]
```

**Bottleneck:** TrungHQ's Phase 4 (fine-tuning) blocks Phase 5. Mitigation: AnhNQ prepares eval harness (5.1) in parallel during tuần 9-10.

---

## Workload Balance

| Member | P1 | P2 | P3 | P4 | P5 | P6 | Total |
|--------|----|----|----|----|----|----|-------|
| TrungHQ | 2 | 4 | 5 | 6 | 4 | 3 | **24** |
| KhoiPND | 3 | 4 | 3 | 2 | 2 | 3 | **17** |
| AnhNQ | 2 | 3 | 5 | 2 | 3 | 3 | **18** |
| ThienTD | 4 | 4 | 3 | 2 | 3 | 4 | **20** |

---

## Risk Mitigation

| Risk | Probability | Impact | Owner | Mitigation |
|------|------------|--------|-------|-----------|
| SLM 7B OOM trên 8GB VRAM | Medium | High | T | Test AWQ/GPTQ early; fallback to 3B model |
| NER accuracy thấp cho Vietnamese | Medium | Medium | A | Hybrid: rule + LLM extraction; manual correction |
| QLoRA không improve | Low | High | T | Increase data, adapter fusion, hyperparameter sweep |
| Ingestion pipeline chậm (220K paragraphs) | Medium | Low | K | Batch processing, parallel workers, progress checkpoints |
| UIT-ViQuAD license concern | Low | Medium | A | Chỉ reformat schema, cite original, không redistribute raw |
| Human verification bottleneck | Medium | Medium | D | Start early (tuần 6), build efficient annotation tool |
| Paper deadline pressure | Medium | High | All | Start draft tuần 10, parallel writing |
