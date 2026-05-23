# Thesis Presentation — Design Spec

## Purpose
Final defense presentation before the thesis committee at FPTU. Updated weekly, final version used for the official defense.

## Parameters
- Duration: 15-20 minutes
- Slides: ~19
- Format: HTML self-contained (ecc:frontend-slides skill)
- Language: English
- Focus: balanced across scientific contribution, technical implementation, and evaluation results

## Slide Structure

### Opening (2 slides)
1. **Title** — thesis title, student name, ID, advisor [placeholder]
2. **Agenda** — presentation outline

### Context & Motivation (3 slides)
3. **Problem** — Vietnamese QA today: single-hop only, hallucination, dependency on external APIs
4. **Objectives** — local, sovereignty-preserving, multi-hop, citation-based KGQA
5. **Contributions** — (1) local GraphRAG system, (2) ViWiki-MHR dataset, (3) fine-tuned SLM

### Architecture & Technical (5 slides)
6. **System Overview** — end-to-end architecture diagram (available)
7. **Knowledge Graph** — schema Page→Chunk→Entity, typed labels, NER pipeline (3 backends)
8. **Retrieval Pipeline** — Cypher generation + safety validation + hybrid fallback + cross-encoder reranking
9. **ReAct Agent** — 4 tools (kg_schema, kg_query, text_search, get_passage), 6-iteration cap, citation tracking
10. **Local SLM** — Qwen2.5-7B-Instruct, 4-bit NF4, Text2Cypher [placeholder: QLoRA results]

### Dataset (3 slides)
11. **Data Sources** — UIT-ViQuAD 2.0 (39.5K, standard benchmark, unanswerable) + ViWiki-MHR (~8K synthetic multi-hop)
12. **ViWiki-MHR Generation** — KG walks (2-hop, 3-hop, broken-link) → Vietnamese templates → LLM rewrite → 3-stage QC
13. **ViQuAD2 Integration** — HF adapter → dedup & ingest → end-to-end eval; role as external benchmark

### Evaluation (2 slides)
14. **Results** — [placeholder: table]
    - Retrieval: Hit Rate, MRR (on ViWiki-MHR)
    - Answer quality: Token F1, EM (on ViQuAD2.0)
    - Abstain accuracy (impossible subset)
    - Baselines: vector-only, BM25-only, graph-only, hybrid
15. **Ablation & Analysis** — [placeholder]
    - Reranking impact
    - NER backend comparison
    - Generative vs extractive mismatch

### Demo (1 slide)
16. **Demo** — sample query → graph traversal → answer with citations [needs screenshot/live]

### Conclusion (3 slides)
17. **Summary & Limitations** — achievements, gaps, limitations
18. **Future Work** — QLoRA, community detection, extended hybrid retrieval
19. **Q&A** — thank the committee

## Data Availability

| Slide | Status |
|-------|--------|
| 1 (title) | placeholder — need student name, ID, advisor |
| 6 (architecture) | available — diagram exists |
| 7-9 (technical) | available — code implemented |
| 10 (SLM) | partial — QLoRA not done |
| 11-13 (dataset) | available — pipeline code done, ViQuAD2 phase 1-2 done |
| 14-15 (eval) | placeholder — awaiting full eval run |
| 16 (demo) | needs screenshot or live demo |

## Style Direction
- Professional, academic tone
- Clean typography, not flashy
- Diagrams as primary visuals
- Clear data tables
- Appropriate for thesis defense context
