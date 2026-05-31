# Research Report: Cải Thiện Hệ Thống GraphRAG Vietnamese Wikipedia

**Date**: 2026-05-27
**Methodology**: 7 parallel research agents, 70+ searches, High effort level

---

## Executive Summary

Hệ thống hiện tại có nhiều điểm có thể cải thiện đáng kể. Dựa trên nghiên cứu 70+ sources từ 2024-2025, xác định **5 trục cải thiện chính** với tổng tiềm năng tăng **+30-50% QA F1** so với baseline hiện tại.

---

## 1. GraphRAG Architectures (SOTA 2024-2025)

| System | Key Innovation | Strengths | Applicable? |
|--------|---------------|-----------|-------------|
| **Microsoft GraphRAG** | Community detection + map-reduce summarization | Global queries, comprehensive answers | Rất phù hợp |
| **LazyGraphRAG** | Deferred indexing, query-time extraction | 0.1% cost of full GraphRAG, comparable quality | Phù hợp cho cost optimization |
| **LightRAG** | Linear scaling, dual-level retrieval | Handles large corpora efficiently | Phù hợp cho 590K articles |
| **Agentic GraphRAG** | RL-based retrieval planning | Adaptive, multi-step reasoning | Future direction |
| **GraphRAG-FI** | Knowledge filtering + integration | Removes noise before retrieval | Rất phù hợp (KG quality issue) |

**Key insight**: Microsoft GraphRAG's community detection approach là missing piece lớn nhất trong hệ thống hiện tại. Nó cho phép trả lời "global questions" mà current system không handle được.

---

## 2. Vietnamese NLP Models

### NER Models (ranked by F1)

| Model | F1 | Type | Notes |
|-------|-----|------|-------|
| NlpHUST/ner-vietnamese-electra-base | 92.14% | Transformer | VLSP 2018 dataset, 4 entity types |
| PhoBERT + Graph Attention | ~91% | Transformer+GNN | Medical domain, 2024 |
| PhoNLP (joint multi-task) | SOTA on VLSP | Multi-task | POS + NER + DEP jointly |
| Hybrid neurosymbolic (2026) | N/A | Rule + DL | Low-resource domains |
| Current system (wikilink) | 46.9% Typed F1 | Rule-based | Wikipedia hyperlinks |

### Embedding Models for Vietnamese

| Model | Dims | Vietnamese Quality | Notes |
|-------|------|-------------------|-------|
| dangvantuan/vietnamese-embedding | 768 | Native, best monolingual | PhoBERT-based |
| BGE-M3 | 1024 | Excellent multilingual | Dense+Sparse+ColBERT unified |
| multilingual-e5-large | 1024 | Good | Microsoft, 100+ langs |
| Gemini embedding (current) | 768 | Good | API-dependent |

### Reranking Models

| Model | Type | Vietnamese Support | Latency (10 docs) |
|-------|------|-------------------|-------------------|
| BAAI/bge-reranker-v2-m3 (current) | Cross-encoder | Multilingual | ~150ms |
| ViRanker | BGE-M3 + BPT | Native Vietnamese | ~120ms |
| Jina-ColBERT-v2 | Late interaction | Multilingual | ~40ms |
| BGE-M3 multi-vector | ColBERT-style | Native multilingual | ~50ms |

---

## 3. Multi-hop QA Techniques

### Taxonomy of Approaches

1. **Question Decomposition**: Tách câu hỏi phức tạp thành sub-queries → giải từng phần → tổng hợp
2. **Iterative Retrieval + Self-Reflection**: Retrieve → assess sufficiency → retrieve more if needed
3. **LLM Planning + Embedding-Guided Search**: LLM lên kế hoạch traversal, embeddings guide path selection
4. **Entity-Centric Summaries**: Pre-compute summaries per entity, retrieve summaries thay vì raw chunks
5. **Subgraph Retrieval**: Extract relevant subgraph rồi reason trên đó

### Most Applicable

- **Iterative retrieval** (extend ReAct agent với self-reflection)
- **Question decomposition** (trước khi gọi Cypher generation)
- **Confidence-weighted traversal** (dùng confidence scores trên edges)

---

## 4. Community Detection & Graph Summarization

### How It Improves Retrieval

Community detection cho phép trả lời "global questions" (span many documents) bằng cách:
- Partition entity graph thành clusters (Leiden algorithm)
- Pre-generate natural language summaries cho mỗi community
- Query time: match question → relevant community summaries → synthesize answer

### Implementation Plan (Neo4j GDS)

```cypher
-- Step 1: Create co-occurrence edges
MATCH (e1:Entity)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:Entity)
WHERE id(e1) < id(e2)
WITH e1, e2, count(c) AS weight
MERGE (e1)-[r:CO_OCCURS]-(e2)
SET r.weight = weight

-- Step 2: Project graph
CALL gds.graph.project('entity-graph', 'Entity', {
  CO_OCCURS: {orientation: 'UNDIRECTED', properties: ['weight']}
})

-- Step 3: Run Leiden at multiple resolutions
CALL gds.leiden.write('entity-graph', {writeProperty: 'community_L0', resolution: 0.2})
CALL gds.leiden.write('entity-graph', {writeProperty: 'community_L1', resolution: 1.0})
CALL gds.leiden.write('entity-graph', {writeProperty: 'community_L2', resolution: 3.0})
```

### Algorithm Comparison

| Feature | Leiden | Louvain | Label Propagation |
|---------|--------|---------|-------------------|
| Quality | Highest | Good (disconnected communities possible) | Lower |
| Speed | Fast O(n log n) | Slightly faster | Fastest |
| Hierarchical | Yes | Yes | No |
| Neo4j GDS | Yes | Yes | Yes |

### Storage Schema

```cypher
CREATE (c:Community {
  id: "community_leiden_L1_42",
  level: 1,
  title: "Triều đại nhà Nguyễn",
  summary: "...",
  summary_embedding: [...],
  entity_count: 47
})
CREATE (e:Entity)-[:BELONGS_TO]->(c:Community)
CREATE (child:Community)-[:PART_OF]->(parent:Community)
```

### Estimated Cost (590K articles)

- Leiden computation: <10 minutes (10M nodes, 50M edges)
- Community summarization: ~$2-8 USD (Gemini Flash)
- Storage overhead: ~150MB
- Memory: 16-32GB for GDS projection

---

## 5. KG Quality & Robustness

### Quantified Impact of KG Issues

| Issue | QA Performance Loss | Current Status |
|-------|--------------------|--------------------|
| Noisy triples (10% noise) | -8-15% F1 | Có (NER errors) |
| Missing links (30% incomplete) | -12-20% F1 | Có (only MENTIONS) |
| Duplicate entities | -5-10% F1 | Có (exact match only) |
| Untyped relations | -3-7% F1 | Có (all MENTIONS) |
| No confidence scores | -5-8% F1 | Có |

### Confidence Scoring Framework

```
confidence(triple) = w1 * source_score + w2 * embedding_score + w3 * frequency_score + w4 * grounding_score
```

- source_score: wikilink=0.9, underthesea=0.7, simple=0.5
- frequency_score: entity mentioned across multiple chunks/pages = higher
- grounding_score: entity appears in chunk text = higher

### Entity Resolution (Ranked by Effectiveness)

1. **Embedding-based clustering** — cosine similarity > 0.85 on entity name embeddings (Tier 1)
2. **Blocking + fuzzy string matching** — Jaro-Winkler after Unicode normalization (Tier 1)
3. **LLM-assisted matching** — For ambiguous cases, compare contexts (Tier 2)
4. **Graph-neighborhood similarity** — Shared neighbors signal duplicates (Tier 2)

### Wikidata Linking Strategy

1. Direct linking via Wikipedia page titles → QID lookup (60-70% coverage)
2. Cross-lingual candidate generation via ParaNames corpus
3. Disambiguation via context embedding similarity
4. Type enrichment from Wikidata P31/P279 properties

### Relationship Extraction

- Few-shot LLM prompting (Gemini) to classify relation types between co-occurring entities
- Relation types: BORN_IN, LOCATED_IN, WORKS_AT, PART_OF, CREATED_BY
- Filter by confidence threshold → store as typed edges

---

## 6. Hybrid Retrieval Architecture

### Pipeline

```
Query → Router → [Vector | Graph | Fulltext] → RRF Fusion → Reranker → LLM
```

### Query Routing Logic

| Query Type | Strategy | Example |
|-----------|----------|---------|
| Factoid (1 entity) | Vector + Fulltext | "Hà Nội thành lập năm nào?" |
| Multi-hop (2+ entities) | Graph + Vector | "Mối quan hệ giữa X và Y?" |
| Exploratory | Vector + Community summaries | "Văn học Việt Nam thế kỷ 20" |
| Aggregation | Graph (Cypher COUNT) | "Bao nhiêu tỉnh ở Việt Nam?" |

### RRF Fusion

```python
score(d) = w_graph * 1/(60 + rank_graph(d))
          + w_vector * 1/(60 + rank_vector(d))
          + w_fulltext * 1/(60 + rank_fulltext(d))
```

Default weights: w_graph=1.2, w_vector=1.0, w_fulltext=0.8

### Neo4j Implementation Patterns

**Pattern 1: Vector + Graph Expansion (single query)**
```cypher
CALL db.index.vector.queryNodes('chunk_embeddings', 20, $query_embedding)
YIELD node AS chunk, score AS vector_score
MATCH (page:Page)-[:HAS_CHUNK]->(chunk)
OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity:Entity)
OPTIONAL MATCH (entity)<-[:MENTIONS]-(related_chunk:Chunk)
RETURN chunk.text, page.title, vector_score, collect(DISTINCT entity.name) AS entities
ORDER BY vector_score DESC LIMIT 10
```

**Pattern 2: Parallel hybrid retrieval**
```python
async def hybrid_retrieve(query, query_embedding):
    vector_task = asyncio.create_task(vector_search(query_embedding, top_k=20))
    fulltext_task = asyncio.create_task(fulltext_search(query, top_k=20))
    graph_task = asyncio.create_task(graph_search(query, top_k=20))

    results = await asyncio.gather(vector_task, fulltext_task, graph_task)
    fused = reciprocal_rank_fusion(results, weights=[1.0, 0.8, 1.2], k=60)
    reranked = await rerank(query, fused[:20])
    return reranked[:10]
```

### Latency Budget (target: <500ms)

| Stage | Target p50 | Target p95 |
|-------|-----------|-----------|
| Query embedding | 15ms | 30ms |
| Query classification | 3ms | 5ms |
| Vector search | 20ms | 50ms |
| Fulltext search | 10ms | 30ms |
| Graph traversal | 30ms | 100ms |
| RRF fusion | 2ms | 5ms |
| Reranking (10 docs) | 80ms | 150ms |
| **Total** | **165ms** | **380ms** |

---

## 7. Evaluation Framework

### Metrics

| Level | Metric | Target |
|-------|--------|--------|
| Retrieval | Recall@10 | >0.85 |
| Retrieval | MRR | >0.70 |
| Retrieval | NDCG@10 | >0.65 |
| Retrieval | Hit Rate@5 | >0.90 |
| Generation | EM (Exact Match) | >0.40 |
| Generation | F1 | >0.60 |
| Generation | Faithfulness (RAGAS) | >0.85 |
| System | Latency p95 | <400ms |

### Benchmarks to Evaluate On

1. **ViWiki-MHR** (custom) — primary evaluation set
2. **VIMQA** — external multi-hop Vietnamese QA benchmark
3. **UIT-ViQuAD 2.0** — Vietnamese MRC (extractive)

### Evaluation Strategy

- Component-wise: NER F1, Retrieval metrics, Generation metrics separately
- End-to-end: RAGAS (faithfulness, relevance, context precision/recall)
- Ablation studies: measure impact of each improvement independently
- A/B testing framework for online comparison

---

## 8. Prioritized Implementation Roadmap

### Phase 1: Quick Wins (Tuần 1-3) — Expected: +15-20% QA F1

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 1.1 | Add confidence scores to MENTIONS edges | 2 days | +5-8% F1 |
| 1.2 | Implement vector search path (Neo4j vector index) | 3 days | +10-15% recall |
| 1.3 | Basic RRF fusion (vector + existing Cypher) | 2 days | +5% MRR |
| 1.4 | Confidence-weighted graph traversal | 1 day | +3-5% F1 |

### Phase 2: Core Improvements (Tuần 4-7) — Expected: +15-25% QA F1

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 2.1 | Embedding-based entity deduplication | 1 week | +5-10% F1 |
| 2.2 | Wikidata linking via Wikipedia page titles | 1 week | +8-12% F1 |
| 2.3 | Full hybrid retrieval (3-path + query router) | 1 week | +10% recall |
| 2.4 | Upgrade reranker (ViRanker or BGE-M3) | 3 days | +5-10% NDCG |
| 2.5 | LLM-based relation extraction (typed edges) | 1 week | +5-10% F1 |

### Phase 3: Advanced Features (Tuần 8-12) — Expected: +10-15% QA F1

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 3.1 | Community detection (Leiden + Neo4j GDS) | 2 weeks | Global query support |
| 3.2 | Community summarization pipeline | 1 week | +comprehensiveness |
| 3.3 | Iterative retrieval with self-reflection | 1 week | +15-20% on multi-hop |
| 3.4 | Question decomposition for complex queries | 1 week | +10% multi-hop |
| 3.5 | Upgrade NER to Electra-base (92% F1) | 1 week | +45% NER F1 |

### Phase 4: Evaluation & Polish (Tuần 13-16)

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 4.1 | RAGAS evaluation pipeline | 3 days | Proper metrics |
| 4.2 | Evaluate on VIMQA benchmark | 1 week | External validation |
| 4.3 | A/B testing framework | 1 week | Rigorous comparison |
| 4.4 | End-to-end latency optimization | 1 week | p95 < 400ms |
| 4.5 | Thesis writeup: ablation studies | 2 weeks | Academic contribution |

---

## 9. Thesis Differentiation Points

1. **Vietnamese-specific GraphRAG**: First application of community detection + hybrid retrieval cho Vietnamese Wikipedia
2. **ViRanker/PhoBERT NER in GraphRAG**: Vietnamese-native models chưa ai apply cho graph-based QA
3. **Hybrid retrieval trên Neo4j**: Vector + Graph + Fulltext fusion với RRF — practical contribution
4. **Ablation study**: Đo impact từng component riêng biệt → clear academic contribution
5. **KG quality impact quantification**: Measure how NER quality, entity dedup, confidence scoring affect downstream QA

---

## 10. Sources

### GraphRAG Architectures
- [Microsoft GraphRAG: From Local to Global](https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/)
- [LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
- [GraphRAG Survey (Jan 2025)](https://arxiv.org/abs/2501.00309)
- [GraphRAG-FI: Knowledge Filtering](https://arxiv.org/html/2503.13804v1)
- [Agentic GraphRAG with RL](https://arxiv.org/abs/2507.21892)
- [LightRAG: Linear Scaling](https://arxiv.org/html/2510.10114v3)

### Vietnamese NLP
- [NlpHUST/ner-vietnamese-electra-base](https://huggingface.co/NlpHUST/ner-vietnamese-electra-base)
- [PhoBERT + Graph Attention for NER](https://arxiv.org/html/2510.11537v1)
- [Vietnamese Massive Text Embedding Benchmark](https://arxiv.org/abs/2507.21500)
- [Hybrid Neurosymbolic Vietnamese NER](https://arxiv.org/abs/2605.04489)
- [Vietnamese Legal Retrieval with Synthetic Data](https://arxiv.org/html/2412.00657v1)
- [PhoNLP: Joint Multi-task Learning](https://ar5iv.labs.arxiv.org/html/2101.01476)
- [dangvantuan/vietnamese-embedding](https://huggingface.co/dangvantuan/vietnamese-embedding)

### Multi-hop QA
- [Robust Multi-Hop GraphRAG Framework](https://arxiv.org/html/2603.14828v1)
- [Entity-Centric Summaries for Multi-hop QA](https://arxiv.org/html/2603.11223)
- [LLM Planning + Embedding-Guided Search](https://arxiv.org/html/2511.19648v1)
- [Hybrid Retrieval on Textual and Relational KBs](https://arxiv.org/abs/2412.16311)
- [Multi-hop QA over KGs using LLMs](http://arxiv.org/abs/2404.19234v1)
- [Structured Prompting and Context Compression](https://arxiv.org/html/2603.14045v1)

### Community Detection
- [Neo4j GDS Leiden Algorithm](https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/)
- [ArchRAG: Attributed Community-based Hierarchical RAG](https://arxiv.org/html/2502.09891v3)
- [Deep GraphRAG: Hierarchical Retrieval](https://arxiv.org/html/2601.11144v2)
- [Maintaining Leiden Communities in Dynamic Graphs](https://arxiv.org/abs/2601.08554)
- [CommunityKG-RAG for Fact-Checking](https://arxiv.org/html/2408.08535v1)
- [LeanRAG: Semantic Aggregation](https://arxiv.org/html/2508.10391v1)

### KG Quality & Robustness
- [Denoising Knowledge Graphs for RAG](https://arxiv.org/html/2510.14271v1)
- [Mitigating Information Loss in KGs](https://arxiv.org/html/2501.15378v1)
- [MERLIN: Multilingual Entity Linking](https://arxiv.org/html/2510.14307)
- [RelatE: Robust KG Embeddings](https://arxiv.org/abs/2505.18971)
- [ParaNames: Entity Names for 400+ Languages](https://arxiv.org/html/2405.09496v1)
- [Triple Confidence Measurement](https://link.springer.com/article/10.1007/s11280-024-01307-x)
- [TRAIL: Joint Inference and Refinement](https://arxiv.org/html/2508.04474)
- [Relations Prediction using LLMs](https://arxiv.org/html/2405.02738v1)

### Hybrid Retrieval
- [HybridRAG: KG + Vector](https://arxiv.org/abs/2408.04948)
- [ViRanker for Vietnamese](https://arxiv.org/abs/2509.09131)
- [BGE-M3 Unified Model](https://arxiv.org/abs/2402.03216)
- [Self-RAG: Iterative Retrieval](https://arxiv.org/abs/2310.11511)
- [Adaptive-RAG: Query Routing](https://arxiv.org/abs/2403.14403)
- [SymRAG: Adaptive Query Routing](https://arxiv.org/html/2506.12981v1)
- [Efficient KG Construction and Hybrid Retrieval](https://arxiv.org/abs/2507.03226)

### Evaluation
- [RAGAS Framework](https://docs.ragas.io/)
- [Component-Based QA Evaluation](https://link.springer.com/chapter/10.1007/978-981-15-5554-1_8)
- [DeepEval End-to-End LLM Evals](https://deepeval.com/docs/evaluation-end-to-end-llm-evals)
- [Comprehensive Evaluation for KGQA](https://arxiv.org/html/2501.17270v1)
