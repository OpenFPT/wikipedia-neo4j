# Handoff Document

**Created:** 2026-06-03
**Topic:** Phase 2 Refactor Complete → Phase 3 Step 1 (Token F1 Quick Wins)
**Status:** In Progress

---

## Context Summary

Completed Phase 2 of the eval-first improvement cycle. Both monolithic files (`agent.py` 911 lines, `hybrid.py` 702 lines) have been split into focused modules. Tests pass (70 green), lint clean. Ready to start Phase 3 Step 1: prompt engineering to boost Token F1 from 0.12 to 0.25+.

---

## Completed Work

- [x] Split `src/orchestration/agent.py` → `agent_loop.py`, `complexity.py`, `decomposition.py`, `voting.py`, `_parsing.py`
- [x] Split `src/retrieval/hybrid.py` → `bm25.py`, `vector.py`, `graph.py`, `community.py`, `fusion.py`
- [x] Created backward-compat shims (`src/agent.py`, `src/retrieve.py`, `src/reranker.py`, `src/neo4j_client.py`)
- [x] Updated tests to patch at correct module paths
- [x] 70 tests passing, 0 lint errors
- [x] Committed as `2340cd6`

---

## Pending Work (Phase 3 Step 1: Free Wins)

- [ ] **Prompt engineering**: Force short-span answers in `SYSTEM_PROMPT` (in `agent_loop.py`)
  - Add instruction: "Answer in as few words as possible, using only words from the context"
  - Target: reduce verbose Vietnamese answers to extractive spans
- [ ] **Post-processing**: Extract minimal answer span from verbose SLM output
  - Add a `_extract_short_answer()` function that strips preamble/filler
  - Apply regex to remove "Dựa trên thông tin..." prefix patterns
- [ ] **Run eval to measure improvement**: `scripts/eval_retrieval_direct.py` + Token F1
- [ ] **Expected improvement**: F1 0.12 → 0.25–0.35

### Phase 3 Step 2 (QLoRA, after Step 1)

- [ ] Write Colab notebook for QLoRA fine-tune (Qwen2.5-7B + ViQuAD2 train)
- [ ] Target: Token F1 > 0.40

### Phase 1 (Ablation, can run anytime)

- [ ] Run `scripts/run_ablation.py` end-to-end
- [ ] Generate `reports/ablation_results.json` and `reports/ablation_table.tex`

---

## Key Decisions Made

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Split into 5+4 modules (not fewer) | Each retrieval signal independently testable/toggleable for ablation | 3-module coarser split |
| Keep original paths via shims | Zero disruption to external consumers (evaluation, MCP, API) | Force all callers to update imports |
| fusion.py remains largest (461 lines) | Contains orchestration + both WRRF implementations + query_graph; further split would create circular deps | Move query_graph to separate module |

---

## Current State

### Key Files
- `src/retrieval/{bm25,vector,graph,community,fusion}.py` — new retrieval modules
- `src/orchestration/{agent_loop,complexity,decomposition,voting,_parsing}.py` — new agent modules
- `src/prompts.py` — prompt templates (target for Phase 3 Step 1 changes)
- `src/evaluation.py:519` — `compute_token_f1()` function
- `scripts/run_ablation.py` — ablation entry point (ready to use)

### Metrics (baseline)
| Metric | Value |
|--------|-------|
| Token F1 | 0.12 |
| Context Hit Rate | 64.7% (200q) |
| MRR | 0.4257 |
| Avg Latency | 125ms |

---

## Next Steps

1. Modify `SYSTEM_PROMPT` in `src/orchestration/agent_loop.py` to instruct short-span answers
2. Add post-processing in `agent_loop.py` to strip verbose preambles from final answers
3. Run eval: `uv run python scripts/eval_retrieval_direct.py --limit 200`
4. If F1 > 0.25, commit. If not, iterate on prompt/postprocessing.
5. Then move to QLoRA notebook for Colab.

---

## Important Notes

- GPU: RTX 4060 8GB (tight for inference, fine-tune on Colab)
- Neo4j: systemd service, `bolt://localhost:7687`
- The `SYSTEM_PROMPT` is in `src/orchestration/agent_loop.py` (line ~29)
- Token F1 is computed in `src/evaluation.py:compute_token_f1()` — Vietnamese-aware tokenization
- The eval script that measures Token F1 for ViQuAD2 is in `src/evaluation.py:evaluate_viquad()`
- Pre-existing broken tests (17 collection errors) due to earlier module moves — not caused by this refactor

---

## Resources

- Design spec: `docs/superpowers/specs/2026-05-31-next-phase-design.md`
- ViQuAD2 data: `data/viquad2/validation.jsonl` (2653 questions with gold answers)
- Eval script: `scripts/eval_retrieval_direct.py`
- Ablation script: `scripts/run_ablation.py`
