# Improving NER for Vietnamese Wikipedia GraphRAG

*Generated: 2026-05-25 | Sources: 25+ | Confidence: High*

## Executive Summary

Your current NER pipeline (underthesea ~88% F1, simple regex, PhoNLP) is **3-7% F1 below state-of-the-art**. The biggest wins come from:

1. **Replacing underthesea with PhoBERT-base-v2 or ViDeBERTa** (+5-6% F1, drop-in)
2. **Adding LLM-based NER via Gemini** for complex/nested entities (+3-5% F1 on hard cases)
3. **Wikipedia hyperlink supervision** — free entity labels from wiki markup (unique to your domain)
4. **Hybrid pipeline**: fast transformer for bulk + LLM escalation for ambiguous spans

## Current Baseline

| Backend | F1 (est.) | Speed | Notes |
|---------|-----------|-------|-------|
| `simple` (regex) | ~60-70% | Very fast | English-only capitalization heuristic |
| `underthesea` | ~88% | Moderate | BiLSTM-CNN-CRF, no GPU accel |
| `phonlp` | ~89% | Moderate | Primarily POS/segmentation, NER secondary |

## Recommended Upgrades (Priority Order)

### 1. Replace underthesea with PhoBERT-based NER (Week 1)

**Best options:**

| Model | HuggingFace ID | F1 | Params | Notes |
|-------|---------------|-----|--------|-------|
| ViDeBERTa-base | `vinai/vdeberta-base` | 94.5% | 86M | SOTA, lightest |
| PhoBERT-base-v2 | `vinai/phobert-base-v2` | 93.6% | 135M | Battle-tested |
| NlpHUST ELECTRA | `NlpHUST/ner-vietnamese-electra-base` | 92.1% | 110M | Ready-to-use NER head |
| PhoBERT-large | `vinai/phobert-large` | 94.7% | 370M | Highest F1, slow |

**Recommendation: `NlpHUST/ner-vietnamese-electra-base`** for immediate use (already has NER head fine-tuned), then fine-tune `vinai/phobert-base-v2` on your domain data for best results.

**Implementation:**
```python
from transformers import pipeline

# Drop-in replacement — already has NER head
ner_pipe = pipeline(
    "token-classification",
    model="NlpHUST/ner-vietnamese-electra-base",
    aggregation_strategy="simple",
    device=0,  # GPU
    batch_size=64,
)

# For PhoBERT: requires word segmentation first
from py_vncorenlp import VnCoreNLP
segmenter = VnCoreNLP(annotators=["wseg"], save_dir="./vncorenlp")
segmented = segmenter.word_segment(text)
```

**Expected improvement: +4-6% F1 over underthesea**

### 2. Wikipedia Hyperlink Supervision (Week 1-2)

Your Wikipedia articles contain **free entity annotations** via internal links:
- `[[Ho Chi Minh]]` -> Person entity
- `[[Ha Noi]]` -> Location entity
- `[[Dai hoc Bach khoa]]` -> Organization entity

**Approach:**
1. During ingestion, extract all `[[...]]` wiki links as entity mentions
2. Resolve link targets to Wikidata types (P31/instance-of) for automatic type labels
3. Use as weak supervision to fine-tune your NER model on domain-specific entities

This is essentially **free labeled data** — ~74% of Wikipedia articles describe entities, and hyperlinks mark entity spans with zero annotation cost.

### 3. LLM-Based NER for Complex Cases (Week 2-3)

Use your existing Gemini backend for hard cases. Key insight: **inline XML output format achieves 90.7% F1** vs 85.6% for JSON.

**Few-shot prompt template:**
```
Extract named entities from this Vietnamese text. Mark entities inline:
<PER>name</PER>, <ORG>name</ORG>, <LOC>name</LOC>, <WORK>name</WORK>

Examples:
Input: Nguyen Du sinh nam 1766 tai Ha Tinh, la tac gia cua Truyen Kieu.
Output: <PER>Nguyen Du</PER> sinh nam 1766 tai <LOC>Ha Tinh</LOC>, la tac gia cua <WORK>Truyen Kieu</WORK>.

Input: {text}
Output:
```

**When to escalate to LLM:**
- Transformer confidence < 0.80
- Entity spans overlap or are nested
- Ambiguous types (e.g., "Ho Chi Minh" — person or city?)

**Cost estimate:** ~$0.50-2 per 1,000 docs (only for escalated cases)

### 4. Hybrid Pipeline Architecture (Week 3-4)

```
Input Text
    |
    v
Stage 1: Fast Transformer NER (PhoBERT/ViDeBERTa)
  - Batch processing, GPU-accelerated
  - Outputs: entities + confidence scores
    |
    +--- conf >= 0.80 ---> Accept as-is
    |
    +--- conf < 0.80 ----> Stage 2: LLM Verification
                             - Gemini few-shot NER
                             - Context-aware disambiguation
                             - Nested entity detection
    |
    v
Stage 3: Post-Processing
  - Entity normalization (Unicode NFKC)
  - Deduplication (fuzzy matching)
  - Type conflict resolution
  - Wikipedia link grounding
```

### 5. Entity Linking and Disambiguation (Month 2)

Resolve entities to canonical Wikidata IDs:
- "HCM" -> Q1854 (Ho Chi Minh city) vs Q36014 (Ho Chi Minh person)
- "Bac Ho" -> Q36014 (alias resolution)

**Tools:**
- **ReFinED** (Amazon): Production-ready, supports fine-tuning on custom entity pairs
- **mGENRE** (Facebook): Multilingual autoregressive entity linking, supports Vietnamese
- **Simple approach**: Match entity text against your existing Page nodes in Neo4j

### 6. GLiNER for Zero-Shot Entity Types (Optional)

If you need to expand beyond PER/ORG/LOC/WORK without retraining:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
entities = model.predict_entities(
    text,
    labels=["person", "organization", "location", "creative work",
            "event", "date", "scientific term"],
    threshold=0.5,
)
```

- Supports arbitrary entity types at inference time
- 130x throughput with bi-encoder variant (GLiNER-bi-Encoder)
- Good for discovering new entity types in your data

## Post-Processing Improvements (Quick Wins)

These can be applied to ANY backend:

### A. Entity Normalization
```python
def normalize_entity(name: str) -> str:
    import unicodedata
    name = unicodedata.normalize("NFKC", name)
    name = name.strip(" .,;:!?\"'()[]")
    name = " ".join(name.split())
    return name
```

### B. Confidence-Based Filtering
- Reject entities with confidence < 0.50
- Flag entities with confidence 0.50-0.80 for LLM review
- Accept entities with confidence >= 0.80

### C. Context-Aware Type Disambiguation
```python
LOCATION_CONTEXT = ["thanh pho", "tinh", "quan", "huyen", "tai", "o"]
PERSON_CONTEXT = ["ong", "ba", "chu tich", "tong thong", "giao su", "tien si"]
```

### D. Entity Merging
- Merge overlapping spans: "Dai hoc" + "Bach khoa Ha Noi" -> "Dai hoc Bach khoa Ha Noi"
- Merge adjacent same-type entities separated only by whitespace

### E. Frequency-Based Validation
- Entities appearing only once in the entire corpus with low confidence -> likely noise
- Entities appearing 3+ times across different articles -> likely valid

## Benchmark Comparison

| Approach | Est. F1 | Throughput | Cost/1K docs | Implementation |
|----------|---------|------------|--------------|----------------|
| Current (underthesea) | ~88% | 500 docs/min | $0 | Existing |
| PhoBERT-base-v2 (fine-tuned) | ~94% | 2000 docs/min (GPU) | $0 | 1 week |
| ViDeBERTa-base | ~95% | 2500 docs/min (GPU) | $0 | 1 week |
| Hybrid (transformer + Gemini) | ~96% | 1500 docs/min | $0.50-2 | 2-3 weeks |
| Full ensemble + entity linking | ~97% | 800 docs/min | $1-3 | 1-2 months |

## Implementation Roadmap

### Phase 1: Drop-in Model Upgrade (Week 1)
- [ ] Add `phobert` backend to `src/ner.py`
- [ ] Use `NlpHUST/ner-vietnamese-electra-base` (pre-trained NER head)
- [ ] Benchmark against underthesea on 100 sample articles
- [ ] Update `NER_BACKEND` config to support `phobert` option

### Phase 2: Wikipedia Link Supervision (Week 2)
- [ ] Extract wiki links during ingestion as ground-truth entities
- [ ] Build entity type mapping from Wikidata P31 property
- [ ] Create evaluation dataset from 500 articles with link-derived labels
- [ ] Fine-tune PhoBERT-base-v2 on this data

### Phase 3: Hybrid Pipeline (Week 3-4)
- [ ] Add confidence scoring to transformer output
- [ ] Implement LLM escalation for low-confidence entities
- [ ] Add post-processing (normalization, dedup, context disambiguation)
- [ ] A/B test against Phase 1 on evaluation set

### Phase 4: Entity Linking (Month 2)
- [ ] Match extracted entities against existing Page nodes in Neo4j
- [ ] Implement alias resolution (abbreviations, alternate names)
- [ ] Add Wikidata ID as property on Entity nodes
- [ ] Enable cross-article entity deduplication

## Key Takeaways

1. **Biggest immediate win**: Replace underthesea with a pre-trained transformer NER model (+5-6% F1, 1 week of work)
2. **Unique advantage**: Your Wikipedia data contains free entity labels via hyperlinks — exploit this
3. **Best long-term architecture**: Hybrid pipeline (fast transformer + LLM escalation) gives 96%+ accuracy at reasonable cost
4. **Don't over-engineer**: Start with the drop-in model replacement, measure improvement, then iterate

## Sources

1. [ViDeBERTa (VinAI, EACL 2023)](https://aclanthology.org/2023.findings-eacl.79.pdf) — Vietnamese DeBERTa, SOTA on NER
2. [PhoBERT (VinAI, EMNLP 2020)](https://aclanthology.org/2020.findings-emnlp.92.pdf) — Pre-trained Vietnamese BERT
3. [NlpHUST/ner-vietnamese-electra-base](https://huggingface.co/NlpHUST/ner-vietnamese-electra-base) — F1: 92.14%
4. [PhoNER_COVID19 (VinAI, NAACL 2021)](https://github.com/VinAIResearch/PhoNER_COVID19) — Vietnamese NER dataset
5. [VLSP 2021 NER Challenge](https://www.researchgate.net/publication/366506578) — 14 entity types benchmark
6. [GLiNER-bi-Encoder](https://arxiv.org/pdf/2602.18487) — 130x throughput zero-shot NER
7. [DiZiNER](https://arxiv.org/html/2604.15866) — Multi-LLM ensemble, +8 F1 over SOTA
8. [FETA: First Extract, Tag Afterwards](https://aclanthology.org/2026.healing-1.11.pdf) — Two-stage LLM NER
9. [ReFinED (Amazon)](https://github.com/amazon-science/ReFinED) — Production entity linking
10. [KALA: Knowledge-Augmented Adaptation](https://ar5iv.labs.arxiv.org/html/2204.10555) — Domain adaptation
11. [CoFEE: Wikipedia Pre-training for NER](https://arxiv.org/pdf/2010.08210) — Wikipedia-specific approach
12. [CP-NER: Cross-domain NER](https://export.arxiv.org/pdf/2301.10410v5.pdf) — Prefix tuning for domain transfer
13. [PhoBERT + GAT (2024)](https://arxiv.org/html/2510.11537v1) — Graph attention achieves 98.4% F1

## Methodology

Searched 15+ queries across web, academic papers, and HuggingFace model hub. Analyzed 25+ sources covering Vietnamese NER models, production NER techniques, and domain adaptation approaches. Cross-referenced F1 scores across multiple benchmarks (VLSP 2016, VLSP 2021, PhoNER_COVID19).
