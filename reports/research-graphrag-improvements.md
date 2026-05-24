# Nghiên cứu phát triển hệ thống GraphRAG cho Wikipedia tiếng Việt

*Ngày: 2026-05-21 | Nguồn: 50+ bài báo & tài liệu (2024-2026) | Thời gian thực hiện: 3 tháng*

---

## Tóm tắt

Hệ thống hiện tại đã có nền tảng tốt (998 pages, 19K chunks, 3.3K entities trong Neo4j). Nghiên cứu chỉ ra 4 hướng phát triển chính để nâng cao chất lượng đồ án:

1. **Cải thiện retrieval** — Thêm cross-encoder reranking (+15.5% accuracy) và query decomposition
2. **Nâng cấp graph construction** — Community detection (Louvain) + entity resolution
3. **Đo lường chất lượng** — RAGAS framework + UIT-ViQuAD 2.0 benchmark
4. **Hybrid retrieval** — Kết hợp graph traversal + vector search + fulltext trong 1 query

---

## 1. GraphRAG State-of-the-Art (2025-2026)

### 1.1 Kiến trúc nổi bật

| Hệ thống | Đặc điểm | Kết quả |
|-----------|-----------|---------|
| **Youtu-GraphRAG** (ICLR 2026) | Agentic paradigm, domain scalable | -33.6% token cost, +16.62% accuracy |
| **StepChain** | Question decomposition + BFS trên KG | +2.57% EM, +2.13% F1 trên HotpotQA |
| **GFM-RAG** | Graph Foundation Model 8M params | +16.8-19.8% trên multi-hop QA |
| **HGRAG** | Hypergraph-based retrieval | 6x speedup retrieval |
| **DualRAG** | Reasoning-augmented + Knowledge Aggregation | Gần oracle performance |

### 1.2 Áp dụng cho project

**Khả thi trong 3 tháng:**
- StepChain approach: decompose câu hỏi → BFS trên graph → merge kết quả
- Community detection (Louvain) để tạo hierarchical summaries
- Hybrid retrieval: graph + vector + fulltext

**Tham khảo:**
- HisGraphRAG (PACLIC 2025): GraphRAG cho lịch sử Việt Nam, dùng sách giáo khoa lớp 12
- ViRAG: Vietnamese RAG với hybrid dense+sparse retrieval
- GraphRAG-Chatbot: Neo4j cho luật Việt Nam

### 1.3 Sources
- [StepChain GraphRAG](https://arxiv.org/abs/2510.02827)
- [GFM-RAG](https://arxiv.org/pdf/2502.01113)
- [HisGraphRAG - PACLIC 2025](https://aclanthology.org/2025.paclic-1.49.pdf)
- [Youtu-GraphRAG - ICLR 2026](https://arxiv.org/abs/2508.19855)
- [ViRAG](https://github.com/anhdao69/ViRAG)

---

## 2. Cải thiện chất lượng câu trả lời

### 2.1 Cross-encoder Reranking (ưu tiên cao nhất)

**Vấn đề hiện tại:** Vector search trả về top-K chunks nhưng thứ tự chưa tối ưu — document đúng có thể ở vị trí 23.

**Giải pháp:** Thêm cross-encoder reranking sau retrieval:
1. Vector search → 50-100 candidates
2. Cross-encoder rerank → top 5-10 chính xác nhất
3. Đưa vào LLM để sinh câu trả lời

**Kết quả kỳ vọng:** +15.5 percentage points accuracy

**Model reranking cho Vietnamese:**
- `BAAI/bge-reranker-v2-m3` (multilingual, hỗ trợ Vietnamese)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (nhẹ, nhanh)

### 2.2 Query Decomposition

**Vấn đề:** Câu hỏi multi-hop phức tạp ("Ai là người sáng lập tổ chức mà Nguyễn Văn A là thành viên?") khó trả lời trong 1 bước.

**Giải pháp:** Tách câu hỏi thành sub-questions:
1. "Nguyễn Văn A là thành viên tổ chức nào?" → query graph
2. "Ai sáng lập tổ chức X?" → query graph
3. Merge kết quả → sinh câu trả lời

### 2.3 Step-back Prompting

Reformulate câu hỏi cụ thể thành câu hỏi tổng quát hơn để tăng recall:
- Gốc: "Hồ Chí Minh sinh năm nào?"
- Step-back: "Thông tin tiểu sử của Hồ Chí Minh"

### 2.4 Faithfulness & Citation

**Phát hiện quan trọng:** 57% citations từ LLM là unfaithful (gắn source sau khi đã quyết định câu trả lời).

**Giải pháp:**
- Validate mỗi claim trong answer phải trace được về chunk cụ thể
- Dùng FRANQ hoặc RAGAS faithfulness metric để đo
- Trong ReAct agent: bắt buộc cite chunk_id cho mỗi statement

### 2.5 Vietnamese-specific

- **Word segmentation:** Dùng RDRsegmenter (ổn định nhất) trước NER và embedding
- **Entity disambiguation:** Tên tiếng Việt nhiều trùng lặp → cần context window rộng hơn
- **Embedding model:** `AITeamVN/Vietnamese_Embedding_v2` hoặc `intfloat/multilingual-e5-small`

### 2.6 Sources
- [Cross-encoder reranking analysis](https://arxiv.org/html/2603.16877v1)
- [Cross-encoder efficiency](https://arxiv.org/html/2405.07920v3)
- [Citation faithfulness](https://arxiv.org/abs/2412.18004)
- [FRANQ](https://arxiv.org/abs/2505.21072v2)
- [Vietnamese word segmentation](https://ar5iv.labs.arxiv.org/html/2301.00418)

---

## 3. Neo4j Advanced Features

### 3.1 Community Detection (Louvain/Leiden)

**Mục đích:** Nhóm các entities liên quan thành communities → tạo summary cho mỗi community → hỗ trợ global search.

**Cách áp dụng:**
1. Chạy Louvain trên graph hiện tại (GDS library)
2. Mỗi community tạo 1 summary text (dùng LLM)
3. Khi query: tìm community liên quan trước → drill down vào entities cụ thể

**Đây là kỹ thuật cốt lõi của Microsoft GraphRAG.**

### 3.2 Neo4j Vector Index (Native)

**Neo4j 5.18+** hỗ trợ vector index native:
```cypher
CREATE VECTOR INDEX chunk_embeddings FOR (c:Chunk) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}

CALL db.index.vector.queryNodes('chunk_embeddings', 10, $queryVector)
YIELD node, score
MATCH (node)<-[:HAS_CHUNK]-(p:Page)
RETURN p.title, node.text, score
```

### 3.3 DRIFT Search Pattern

Dynamic Reasoning and Inference with Flexible Traversal:
1. Vector search trên community summaries
2. LLM sinh follow-up queries
3. Parallel local search trên entities
4. Iterative deepening (max depth = 2)

### 3.4 Performance Optimization

| Kỹ thuật | Cải thiện |
|----------|-----------|
| Block Format Storage (Neo4j 5.14+) | +40-70% performance |
| Token lookup indexes | Bắt buộc cho label/type predicates |
| Query result caching (LRU, 500 queries, 5min TTL) | <1ms cho cached queries |
| Parameterized Cypher | Tái sử dụng execution plan |

### 3.5 Sources
- [Neo4j Vector Indexes](https://neo4j.com/docs/cypher-manual/5/indexes/semantic-indexes/vector-indexes/)
- [DRIFT Search with Neo4j](https://neo4j.com/blog/developer/drift-search-with-neo4j-and-llamaindex/)
- [Neo4j Block Format Storage](https://neo4j.com/blog/developer/neo4j-graph-native-store-format/)
- [Hybrid Retrieval for GraphRAG](https://medium.com/neo4j/hybrid-retrieval-for-graphrag-applications-using-the-neo4j-genai-python-package-fddfafe06ff3)

---

## 4. Đo lường chất lượng hệ thống

### 4.1 Benchmark cho Vietnamese QA

| Dataset | Mô tả | Metrics |
|---------|--------|---------|
| **UIT-ViQuAD 2.0** | 35,990 QA pairs từ Wikipedia, có unanswerable | F1, EM |
| **ViMMRC 2.0** | 5,273 multiple-choice từ văn học | Accuracy |
| **VLUE** | 5 tasks NLU tiếng Việt | Composite score |

**Baseline (VLUE leaderboard):**
- CafeBERT: 77.51% EM, 65.25% F1
- XLM-RoBERTa-large: 76.53% EM, 64.71% F1
- PhoBERT-large: 73.07% EM, 57.27% F1

### 4.2 RAG Evaluation Framework: RAGAS

**Metrics chính (không cần ground truth):**

| Metric | Đo gì | Cách tính |
|--------|--------|-----------|
| **Faithfulness** | Answer có hallucinate không? | Mỗi claim → check có trong context không |
| **Answer Relevancy** | Answer có trả lời đúng câu hỏi? | Sinh reverse questions → so sánh similarity |
| **Context Precision** | Context retrieved có chính xác? | Relevant chunks ở top hay bottom? |
| **Context Recall** | Có thiếu thông tin quan trọng? | Ground truth claims → check coverage |

### 4.3 Evaluation Pipeline đề xuất

```
                    ┌─────────────────────────────────┐
                    │   Test Set (200-500 questions)   │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │     GraphRAG System (query)      │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
    ┌─────────▼─────────┐ ┌───────▼───────┐ ┌─────────▼─────────┐
    │   Retrieval Eval   │ │ Generation Eval│ │   Graph Quality   │
    │ - Context Precision│ │ - Faithfulness │ │ - Entity coverage │
    │ - Context Recall   │ │ - Relevancy    │ │ - Relation F1     │
    │ - MRR, NDCG        │ │ - F1, EM       │ │ - Completeness    │
    └────────────────────┘ └───────────────┘ └───────────────────┘
```

### 4.4 Tạo Test Set

**Nguồn 1:** Lấy subset từ UIT-ViQuAD 2.0 (các câu hỏi liên quan đến pages đã ingest)

**Nguồn 2:** Dùng dataset_gen.py hiện có:
- KG walk → template questions → LLM rewrite
- Đã có sẵn ground truth (answer từ graph)
- Thêm human validation cho 50-100 câu

**Nguồn corpus:** dùng snapshot Vietnamese Wikipedia đã publish trên Hugging Face:
`Keithsel/viwiki-20260523`. Snapshot này được tạo từ raw MediaWiki XML bằng
`scripts/viwiki_processing/`, xuất cả `articles_cleaned/` cho ingestion/retrieval
và `articles_raw/` để giữ wikitext phục vụ link/template extraction.

**Nguồn 3:** Tự tạo multi-hop questions:
- 2-hop: "Ai sáng lập tổ chức X?" (cần traverse 2 edges)
- 3-hop: "Thành phố nào là nơi sinh của người sáng lập X?" (3 edges)
- Comparison: "So sánh A và B về thuộc tính Y"

### 4.5 Sources
- [UIT-ViQuAD 2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0)
- [ViWiki 2026-05-23 snapshot](https://huggingface.co/datasets/Keithsel/viwiki-20260523)
- [VLUE Benchmark](https://uitnlpgroup.github.io/VLUE/)
- [RAGAS Framework](https://arxiv.org/html/2309.15217v1)
- [DeepEval RAG Evaluation](https://deepeval.com/guides-rag-evaluation)

---

## 5. Kế hoạch thực hiện 3 tháng

### Tháng 1: Foundation & Evaluation Setup

| Tuần | Công việc | Output |
|------|-----------|--------|
| 1 | Setup RAGAS evaluation pipeline | `src/evaluation.py` chạy được |
| 1 | Tạo test set 200 câu (mix auto + manual) | `data/test_set.jsonl` |
| 2 | Baseline evaluation trên hệ thống hiện tại | Báo cáo baseline metrics |
| 2 | Thêm cross-encoder reranking | `src/reranker.py` |
| 3 | Implement query decomposition | Update `src/retrieve.py` |
| 3 | Đo lại metrics sau improvements | So sánh trước/sau |
| 4 | Community detection (Louvain) trên graph | Communities trong Neo4j |
| 4 | Tạo community summaries | Summary nodes trong graph |

### Tháng 2: Advanced Features & Optimization

| Tuần | Công việc | Output |
|------|-----------|--------|
| 5 | Neo4j vector index (native) | Hybrid retrieval hoạt động |
| 5 | DRIFT search pattern (simplified) | Multi-level retrieval |
| 6 | Entity resolution & deduplication | Cleaner graph |
| 6 | Improve NER (RDRsegmenter + better classification) | Better entity extraction |
| 7 | Faithfulness validation trong answer generation | Citation tracking |
| 7 | Expand test set (thêm multi-hop questions) | 500 câu test |
| 8 | Full evaluation round 2 | Metrics comparison report |

### Tháng 3: Polish & Documentation

| Tuần | Công việc | Output |
|------|-----------|--------|
| 9 | Performance optimization (caching, indexing) | Faster response time |
| 9 | Ingest thêm data (target 5000+ pages) | Larger knowledge graph |
| 10 | Final evaluation round 3 | Final metrics |
| 10 | A/B comparison: naive RAG vs GraphRAG | Comparison report |
| 11 | Viết báo cáo đồ án (methodology, results) | Draft báo cáo |
| 12 | Demo, presentation prep | Slides + demo video |

### Metrics mục tiêu

| Metric | Baseline (ước tính) | Target |
|--------|---------------------|--------|
| Faithfulness | ~0.6 | >=0.85 |
| Answer Relevancy | ~0.5 | >=0.75 |
| Context Precision | ~0.4 | >=0.70 |
| Multi-hop F1 | ~0.3 | >=0.55 |
| Response time (p95) | ~5s | <=3s |

---

## 6. Đề xuất ưu tiên (Impact vs Effort)

```
High Impact, Low Effort (LÀM TRƯỚC):
├── Cross-encoder reranking (+15.5% accuracy)
├── RAGAS evaluation setup (đo được = cải thiện được)
└── Neo4j vector index (native, thay thế external)

High Impact, Medium Effort:
├── Query decomposition cho multi-hop
├── Community detection + summaries
└── Test set creation (200-500 câu)

Medium Impact, Medium Effort:
├── DRIFT search pattern
├── Entity resolution
└── Faithfulness validation

Lower Priority (nếu còn thời gian):
├── Expand data (5000+ pages)
├── Vietnamese embedding model fine-tune
└── Step-back prompting
```

---

## 7. Tài liệu tham khảo chính

### Papers
1. StepChain GraphRAG (arXiv:2510.02827) — Question decomposition + BFS
2. GFM-RAG (arXiv:2502.01113) — Graph Foundation Model for RAG
3. HisGraphRAG (PACLIC 2025) — GraphRAG cho lịch sử Việt Nam
4. Youtu-GraphRAG (ICLR 2026) — Agentic GraphRAG
5. RAGAS (arXiv:2309.15217) — RAG evaluation framework
6. Citation Faithfulness (arXiv:2412.18004) — 57% citations unfaithful

### Datasets
7. UIT-ViQuAD 2.0 — Vietnamese QA benchmark
8. VLUE — Vietnamese Language Understanding Evaluation

### Tools & Libraries
9. Neo4j GDS — Community detection, embeddings
10. Neo4j Vector Index — Native vector search
11. RAGAS Python — Evaluation metrics
12. BGE-reranker-v2-m3 — Multilingual cross-encoder

### Vietnamese NLP
13. RDRsegmenter — Word segmentation
14. PhoBERT — Vietnamese BERT
15. AITeamVN/Vietnamese_Embedding_v2 — Vietnamese embeddings
