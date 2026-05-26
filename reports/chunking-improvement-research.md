# Improving Chunking for GraphRAG: Research Report

*Generated: 2026-05-25 | Sources: 30+ | Confidence: High*

## Executive Summary

Your current chunking (`chunk_text()`) is a fixed-size character splitter (900 chars, ~225 tokens, 120-char overlap) with no sentence boundary awareness. Research shows this is suboptimal for both retrieval accuracy and entity extraction quality. The highest-impact improvements are:

1. **Sentence-aligned boundaries** (immediate, zero-cost) — prevents mid-word/mid-sentence cuts
2. **Increase chunk size to 400-512 tokens** (~1600-2048 chars for Vietnamese) — research consensus for best recall
3. **Wikipedia section-aware splitting** (structure-first, then size) — paragraph grouping achieved 0.459 nDCG@5 vs 0.244 for fixed-character splitting
4. **Drop overlap or reduce to minimal** — systematic studies show overlap provides no measurable retrieval benefit
5. **Add contextual metadata** (section title prepended) — Anthropic's contextual retrieval showed 67% reduction in retrieval failures

## 1. Current Implementation Analysis

```python
# src/text_utils.py — current implementation
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    # Fixed character window, no boundary awareness
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append(cleaned[i:j])
```

**Problems identified:**
- 900 chars ≈ 225 tokens — below the 400-512 token sweet spot found in benchmarks
- Cuts mid-sentence and mid-word (Vietnamese multi-syllable words use spaces: "Ho Chi Minh" could split to "Ho Chi" | "Minh")
- Destroys Wikipedia section structure (headings, paragraphs)
- 120-char overlap adds indexing cost with no proven retrieval benefit
- Entity mentions split across boundaries degrade NER-chunk linking

## 2. Recommended Chunking Strategy

### Tier 1: Structure-Aware Recursive Splitting (Recommended)

Based on the research, the best approach for Wikipedia articles is a **hierarchical recursive splitter** that respects document structure:

```
Priority order of split points:
1. Wikipedia section headings (== Heading ==)
2. Paragraph boundaries (\n\n)
3. Sentence boundaries (Vietnamese sentence-ending punctuation: .!?)
4. Word boundaries (spaces)
5. Character boundary (last resort)
```

**Target parameters:**
- Chunk size: **1600 characters** (~400 tokens for Vietnamese text)
- Max chunk size: **2048 characters** (~512 tokens)
- Overlap: **0** (or 1 sentence if needed for coreference)
- Minimum chunk size: **200 characters** (merge small sections into next)

**Why these numbers:**
- 512 tokens achieved 69% accuracy (best) in Vecta benchmark on academic papers ([runvecta.com](https://www.runvecta.com/blog/we-benchmarked-7-chunking-strategies-most-advice-was-wrong))
- Recursive-512 achieved 71% recall@5 vs semantic's 68.5% ([abhilashganji.com](https://abhilashganji.com/research/rag-chunking-strategies.html))
- Paragraph grouping achieved 0.459 nDCG@5 — nearly 2x fixed-character's 0.244 ([arXiv 2603.06976](https://arxiv.org/pdf/2603.06976))
- "Chunk size matters more than strategy. Getting from 128 to 512 tokens improved recall more than switching strategies" ([abhilashganji.com](https://abhilashganji.com/research/rag-chunking-strategies.html))

### Tier 2: Contextual Metadata Enrichment

Prepend section context to each chunk before embedding:

```
[Page: Ho Chi Minh | Section: Tieu su > Thoi nien thieu]
Nguyen Sinh Cung sinh ngay 19 thang 5 nam 1890...
```

This is based on Anthropic's "Contextual Retrieval" approach which reduced retrieval failures by 67% when combined with reranking ([DEV Community, 2026](https://dev.to/saurabh_naik_b213f3bbeafe/chunking-for-rag-stop-tuning-the-wrong-knob-3mke)).

### Tier 3: Entity-Aware Boundary Adjustment

After determining chunk boundaries, adjust to avoid splitting named entities:

1. Run NER on the full text first (already done in your pipeline)
2. If a chunk boundary falls inside an entity span, extend the chunk to include the full entity
3. This prevents "Nguyen Ai" | "Quoc" splits

## 3. Why NOT Semantic Chunking

Research strongly suggests semantic chunking is **not worth the cost** for your use case:

- NAACL 2025 finding: "fixed-size chunking often outperformed on real-world documents" — semantic chunking benefits were inconsistent ([aclanthology.org/2025.findings-naacl.114.pdf](https://aclanthology.org/2025.findings-naacl.114.pdf))
- Semantic chunking produced average 43-token chunks in Vecta benchmark — too fragmented, collapsed on document-level metrics (54% accuracy vs recursive's 69%)
- Processing time: 45 minutes vs 15 seconds for recursive on same corpus
- Vietnamese embedding models may not produce reliable similarity scores for boundary detection
- Your reranker (BGE-reranker-v2-m3) already closes the gap: "reranker compressed strategy gaps by ~50%" ([abhilashganji.com](https://abhilashganji.com/research/rag-chunking-strategies.html))

**Exception:** If you later find specific failure modes where coherent passages are being split, consider semantic chunking as a targeted fix, not a wholesale replacement.

## 4. Vietnamese-Specific Considerations

### Sentence Boundary Detection

Vietnamese sentence boundaries are marked by standard punctuation (`.`, `!`, `?`) but have complications:
- Abbreviations: "TP.", "PGS.", "TS." — need abbreviation list
- Quoted speech within sentences
- Numbered lists within paragraphs

**Recommended approach:** Use regex with Vietnamese abbreviation exceptions:

```python
# Vietnamese sentence boundary pattern
VIET_ABBREVS = {"TP", "PGS", "TS", "GS", "ThS", "CN", "Bs", "KTS", "TSKH"}
```

### Word Boundary Awareness

Vietnamese words are space-separated syllables. Multi-syllable words (proper nouns especially) should not be split:
- "Ho Chi Minh" (3 syllables, 1 entity)
- "Thanh pho Ho Chi Minh" (5 syllables, 1 entity)

The entity-aware boundary adjustment (Tier 3) handles this naturally.

### Wikipedia Section Structure

Vietnamese Wikipedia uses standard MediaWiki markup:
```
== Heading 2 ==
=== Heading 3 ===
==== Heading 4 ====
```

These are natural, high-quality split points that align with topic boundaries.

## 5. Implementation Design

```python
def chunk_text_v2(
    text: str,
    title: str = "",
    max_chunk_size: int = 2048,    # ~512 tokens
    target_chunk_size: int = 1600,  # ~400 tokens
    min_chunk_size: int = 200,
    include_context: bool = True,
) -> list[dict[str, str]]:
    """
    Structure-aware recursive chunking for Vietnamese Wikipedia.
    
    Returns list of {"text": ..., "context": ..., "section": ...}
    """
```

**Algorithm:**
1. Parse Wikipedia section structure (split on `== ... ==` patterns)
2. For each section, split on paragraph boundaries (`\n\n`)
3. If a paragraph exceeds `max_chunk_size`, split on sentence boundaries
4. If a sentence exceeds `max_chunk_size` (rare), split on word boundaries
5. Merge consecutive small chunks (< `min_chunk_size`) within same section
6. Prepend section path as context metadata

**Output format change:**
```python
# Before: list[str]
["chunk text 1", "chunk text 2", ...]

# After: list[dict]
[
    {"text": "chunk text", "section": "Tieu su > Thoi nien thieu", "page_title": "Ho Chi Minh"},
    ...
]
```

## 6. Impact on Downstream Pipeline

### Embedding Quality
- Larger, coherent chunks produce better embeddings (less noise, more semantic signal)
- Section context in embedding input improves retrieval relevance

### NER Accuracy
- Entity spans no longer split across chunks
- Section context helps disambiguate entities (same name in different sections)

### Graph Construction
- `MENTIONS` edges become more precise (entity clearly within chunk boundary)
- Section metadata enables richer graph structure: `Page -[:HAS_SECTION]-> Section -[:HAS_CHUNK]-> Chunk`

### Retrieval
- Fewer, larger chunks = fewer candidates to search = faster retrieval
- Better chunk coherence = less noise in reranking
- Section metadata enables filtered retrieval (search within specific sections)

### Breaking Changes
- Chunk IDs will change (different boundaries = different content = different UUIDs)
- Requires full re-ingestion of the dataset
- Embedding dimensions unchanged, but all embeddings need regeneration

## 7. Evaluation Plan

### Metrics to Track

| Metric | What it measures | Target |
|--------|-----------------|--------|
| Recall@5 | Relevant chunks in top-5 | > 70% |
| MRR@10 | Rank of first relevant chunk | > 0.5 |
| nDCG@5 | Graded relevance in top-5 | > 0.45 |
| Avg chunk tokens | Chunk size distribution | 350-500 |
| Entity split rate | % entities cut at boundaries | < 5% |
| Boundary coherence | Chunks end at sentence boundaries | > 95% |

### A/B Testing Approach

1. Build eval set: 50-100 questions from ViWiki-MHR benchmark
2. Baseline: current chunking (900 chars, 120 overlap)
3. Variant A: recursive-1600 (no overlap, sentence-aligned)
4. Variant B: recursive-1600 + section context prepended
5. Variant C: recursive-1600 + section context + entity-aware boundaries
6. Measure recall@5, MRR@10 with and without reranker

**Important:** Equalize context budgets when comparing. Different chunk sizes need different k values to provide equivalent context to the LLM ([runvecta.com](https://www.runvecta.com/blog/we-benchmarked-7-chunking-strategies-most-advice-was-wrong)).

### Tools

- **ChunkViz** ([chunkviz.up.railway.app](https://chunkviz.up.railway.app/)) — visualize chunk boundaries
- **MTCB** ([pypi.org/project/mtcb](https://pypi.org/project/mtcb/)) — standardized chunking benchmark
- Your existing `src/evaluation.py` — context hit rate, MRR, latency

## 8. Migration Path

### Phase 1: Quick Win (1-2 days)
- Implement sentence-aligned splitting with increased chunk size (1600 chars)
- Drop overlap
- Run evaluation against baseline

### Phase 2: Structure-Aware (3-5 days)
- Add Wikipedia section parsing
- Implement hierarchical recursive splitting
- Add section metadata to chunk output
- Update `export_dataset.py` and `load_neo4j.py` for new format

### Phase 3: Context Enrichment (2-3 days)
- Prepend section path to chunks before embedding
- Add `Section` nodes to graph schema
- Update retrieval queries to leverage section metadata

### Phase 4: Entity-Aware Boundaries (2-3 days)
- Integrate NER span information into boundary decisions
- Validate entity split rate < 5%
- Full re-ingestion and evaluation

## Key Takeaways

1. **Chunk size matters more than strategy** — moving from 225 to 400-512 tokens will likely give the biggest single improvement
2. **Structure-aware splitting is the sweet spot** — paragraph grouping nearly doubles nDCG vs fixed-character, with negligible compute cost
3. **Skip semantic chunking** — your reranker already compensates, and the cost/complexity isn't justified
4. **Drop overlap** — systematic studies show no retrieval benefit, just indexing cost
5. **Section context is cheap and powerful** — prepending "Page: X | Section: Y" before embedding is a high-ROI change

## Sources

1. [Is Semantic Chunking Worth the Computational Cost? — NAACL 2025](https://aclanthology.org/2025.findings-naacl.114.pdf)
2. [Chunking for RAG: Benchmarking 36 Methods — arXiv 2603.06976](https://arxiv.org/pdf/2603.06976)
3. [We Benchmarked 7 Chunking Strategies — Vecta Blog 2026](https://www.runvecta.com/blog/we-benchmarked-7-chunking-strategies-most-advice-was-wrong)
4. [RAG Chunking Strategies Benchmark — Abhilash Ganji 2025](https://abhilashganji.com/research/rag-chunking-strategies.html)
5. [Chunk Size and Embedding Model Study — arXiv 2505.21700](https://www.arxiv.org/pdf/2505.21700)
6. [Mix-of-Granularity (MoG) — COLING 2025](https://aclanthology.org/2025.coling-main.384.pdf)
7. [Document Segmentation Matters — ACL 2025 Findings](https://aclanthology.org/2025.findings-acl.422.pdf)
8. [LumberChunker — EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.377.pdf)
9. [TopoChunker — arXiv 2603.18409](https://arxiv.org/abs/2603.18409v1)
10. [Adaptive Chunking with Intrinsic Metrics — arXiv 2603.25333](https://arxiv.org/pdf/2603.25333)
11. [Recursive Semantic Chunking — ICNLSP 2025](https://aclanthology.org/2025.icnlsp-1.15.pdf)
12. [Systematic Analysis of Chunking — ECIR 2026](https://link.springer.com/chapter/10.1007/978-3-032-21321-1_9)
13. [Chunking for RAG: Stop Tuning the Wrong Knob — DEV Community 2026](https://dev.to/saurabh_naik_b213f3bbeafe/chunking-for-rag-stop-tuning-the-wrong-knob-3mke)
14. [Azure AI Search: Chunk by Document Layout](https://www.learn.microsoft.com/en-us/azure/search/search-how-to-semantic-chunking)
15. [ChunkViz — Visualization Tool](https://chunkviz.up.railway.app/)
16. [MTCB: Massive Text Chunking Benchmark](https://pypi.org/project/mtcb/)
17. [ChunkRAG: LLM-Driven Chunk Filtering — arXiv 2410.19572](https://arxiv.org/html/2410.19572v3)
18. [W-RAC: Web Retrieval-Aware Chunking — arXiv 2604.04936](https://www.arxiv.org/pdf/2604.04936)

## Methodology

Searched 25+ queries across academic databases (ACL Anthology, arXiv), engineering blogs, and framework documentation. Analyzed 30+ sources covering semantic chunking, graph-aware chunking, multilingual considerations, and evaluation methods. Cross-referenced benchmark results across multiple independent studies.
