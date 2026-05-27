# Mô tả đồ án — Vietnamese GraphRAG: Hệ thống Hỏi-Đáp đồ thị tri thức trên Wikipedia tiếng Việt

## 1. Tổng quan đề tài

**Tên đề tài:** Vietnamese GraphRAG — Knowledge Graph Question Answering over Wikipedia with a Local Small Language Model and Neo4j

**Tên tiếng Việt:** Hệ thống hỏi-đáp đa bước trên đồ thị tri thức Wikipedia tiếng Việt sử dụng mô hình ngôn ngữ nhỏ chạy cục bộ và Neo4j

**Mô tả ngắn:** Xây dựng hệ thống GraphRAG hoàn toàn cục bộ (local), cho phép trả lời câu hỏi đa bước (multi-hop) trên nội dung Wikipedia tiếng Việt bằng cách kết hợp đồ thị tri thức Neo4j, mô hình ngôn ngữ nhỏ (SLM) lượng tử hóa 4-bit, và cơ chế trích dẫn nguồn xác minh được.

---

## 2. Vấn đề nghiên cứu

Hệ thống hỏi-đáp tiếng Việt hiện tại có **3 hạn chế chính**:

### 2.1. Chỉ hỗ trợ câu hỏi đơn bước (single-hop)

- Các hệ thống hiện có (ViQuAD, PhoBERT-QA) chỉ trích xuất câu trả lời từ **một đoạn văn duy nhất**.
- Không thể trả lời câu hỏi cần suy luận qua nhiều bước, ví dụ: *"Ai sáng lập tổ chức có trụ sở tại Hà Nội?"* — cần tra cứu tổ chức → tìm người sáng lập.

### 2.2. Ảo giác (Hallucination)

- Nghiên cứu của Liu et al. cho thấy **57% trích dẫn do LLM tạo ra là không trung thực**.
- Không có cơ chế xác minh nguồn → người dùng không thể tin tưởng câu trả lời.

### 2.3. Phụ thuộc API bên ngoài

- Sử dụng GPT-4 hoặc Claude đồng nghĩa với việc gửi dữ liệu tiếng Việt ra server nước ngoài.
- Vấn đề: chi phí cao, độ trễ lớn, **chủ quyền dữ liệu** (data sovereignty) cho tổ chức cần triển khai nội bộ.

**Kết luận:** Chưa tồn tại hệ thống KGQA (Knowledge Graph Question Answering) tiếng Việt hoàn toàn cục bộ nào.

---

## 3. Mục tiêu nghiên cứu

| # | Mục tiêu | Sản phẩm cụ thể | Chỉ số đo lường |
|---|----------|-----------------|-----------------|
| O1 | Xây dựng hệ thống GraphRAG cục bộ | Dịch vụ QA chạy trên phần cứng phổ thông | 8GB VRAM, < 3 giây latency |
| O2 | Tạo bộ dữ liệu đa bước mở | ViWiki-MHR (~36K mẫu, 6 loại suy luận) | CC-BY-SA, có gold Cypher |
| O3 | Đảm bảo trích dẫn nguồn | Citation verifier xác minh mỗi câu trả lời | 100% câu trả lời có passage_id |
| O4 | Đánh giá trên benchmark chuẩn | Kết quả trên ViQuAD 2.0 + ViWiki-MHR | Token F1 > 0.40, Hit Rate > 0.60 |

**Ràng buộc chính:** Mọi thứ chạy cục bộ, không gọi API bên ngoài.

---

## 4. Kiến trúc hệ thống

### 4.1. Luồng xử lý tổng quan

```
Câu hỏi người dùng
    → SLM Orchestrator (Vi-Qwen2-7B-RAG, 4-bit NF4, ~5.5GB VRAM)
    → Phát hiện độ phức tạp (complexity detection)
    → Vòng lặp ReAct (tối đa 6 bước) hoặc Question Decomposition + Multi-trajectory
    → 4 nguồn truy xuất: BM25 | Vector | Graph Traversal | Community (WRRF fusion)
    → Cross-encoder Reranker + Citation Verifier
    → Câu trả lời cuối cùng + nguồn trích dẫn
```

### 4.2. Sáu công cụ (Tools) của Agent

| Tool | Chức năng | Giải thích |
|------|-----------|------------|
| `kg_schema()` | Trả về schema đồ thị (cached) | Agent biết đồ thị có những node/relation nào để sinh Cypher đúng |
| `kg_query(cypher)` | Thực thi Cypher trên Neo4j | Truy vấn có cấu trúc, trả về kết quả hoặc lỗi để retry |
| `text_search(query, k)` | Tìm kiếm hybrid BM25 + dense | Fallback khi Cypher thất bại hoặc KG thiếu thông tin |
| `get_passage(id)` | Lấy đoạn văn gốc | Dùng cho citation — xác minh câu trả lời có căn cứ |
| `entity_neighborhood(entity, hops)` | Lấy thực thể lân cận | Khám phá quan hệ xung quanh một entity trong k bước |
| `path_search(entity_a, entity_b, max_hops)` | Tìm đường đi ngắn nhất | Tìm quan hệ giữa 2 thực thể qua đồ thị |

### 4.3. Giải thích cơ chế hoạt động

**ReAct loop (Reason + Act):**
1. Agent **suy nghĩ** (Thought): phân tích câu hỏi, quyết định cần tool nào.
2. Agent **hành động** (Action): gọi tool, nhận kết quả (Observation).
3. Lặp lại cho đến khi đủ bằng chứng hoặc hết 6 bước.
4. Nếu KG trả về rỗng → chuyển sang `text_search` hoặc **từ chối trả lời** (abstain).

**Tại sao giới hạn 6 bước?** Tránh vòng lặp vô hạn khi model bị kẹt. Thực tế, câu hỏi 1-2 hop chỉ cần 2-3 bước.

---

## 5. Đồ thị tri thức (Knowledge Graph)

### 5.1. Schema

```
Page -[:HAS_CHUNK]-> Chunk -[:MENTIONS]-> Entity
Page -[:LINKS_TO]-> Page
```

**Loại Entity:** `Person`, `Organization`, `Location`, `Work`

**Cạnh có kiểu:** `MENTIONS_PERSON`, `MENTIONS_ORG`, `MENTIONS_LOCATION`, `MENTIONS_WORK`

### 5.2. Quy mô hiện tại

| Thành phần | Số lượng | Ghi chú |
|-----------|---------|---------|
| Page (bài viết) | 998 | Prototype, mục tiêu 5000+ |
| Chunk (đoạn văn) | ~19,000 | Mỗi bài chia thành nhiều chunk |
| Entity (thực thể) | ~3,300 | Trích xuất bằng NER |

### 5.3. NER pluggable (6 backend)

| Backend | Cơ chế | Ưu/nhược |
|---------|--------|----------|
| `simple` | Regex + keyword classification | Nhanh, không cần model, độ chính xác thấp |
| `underthesea` | BIO tagging (CRF/BiLSTM) | Offline, tốt cho tiếng Việt, cần cài thêm |
| `phonlp` | PhoNLP + VnCoreNLP word segmentation | Chính xác nhất, nặng nhất |
| `phobert` | PhoBERT transformer pipeline | Chính xác cao, cần GPU |
| `videberta` | ViDeBERTa (NlpHUST electra-base) | Tốt cho NER tiếng Việt, cần GPU |
| `wikilink` | Wikipedia hyperlinks | Tốt nhất cho bulk ingestion (F1=46.9%), không cần model |

**Tại sao pluggable?** Cho phép chọn backend phù hợp với tài nguyên máy. Prototype dùng `simple`, bulk ingestion dùng `wikilink`, production dùng `phobert` hoặc `videberta`.

---

## 6. Pipeline Text2Cypher

Chuyển câu hỏi tiếng Việt thành truy vấn Cypher an toàn qua **4 giai đoạn**:

### Giai đoạn 1: Schema Linking
- Dùng BM25/TF-IDF để **lọc schema** — chỉ giữ lại label và relation liên quan đến câu hỏi.
- Giải thích: Schema đầy đủ quá lớn cho context window của SLM, cần prune.

### Giai đoạn 2: SLM sinh Cypher
- Model nhận câu hỏi + schema đã lọc → sinh câu truy vấn Cypher.
- Ví dụ: *"Năm sinh của tác giả Tắt đèn?"* → `MATCH (w:Work {name:"Tắt đèn"})<-[:SÁNG_TÁC]-(p:Person) RETURN p.birth_year`

### Giai đoạn 3: CyVer Validation (xác minh xác định)
- Kiểm tra **cú pháp** (syntax), **schema alignment** (có đúng label/relation không), **bảo mật** (chặn mọi keyword ghi: CREATE, DELETE, SET...).
- Đây là bước **deterministic** — không dùng AI, chỉ dùng rule-based.

### Giai đoạn 4: Error Refinement Loop
- Nếu validation thất bại → lỗi được đưa lại cho SLM như observation → SLM sinh lại Cypher.
- Lấy cảm hứng từ CyVerACT (2024).

---

## 7. Hybrid Retrieval & Reranking

### Tại sao cần hybrid?

| Phương pháp | Ưu điểm | Nhược điểm |
|-------------|---------|------------|
| Graph-only (Cypher) | Chính xác cho quan hệ có cấu trúc | Thất bại khi KG thưa (sparse) |
| Text-only (BM25 + Dense) | Bao phủ rộng | Không suy luận đa bước |
| **Hybrid (của chúng tôi)** | Graph-first, text fallback | Phức tạp hơn nhưng robust |

### Pipeline 6 bước:
1. BM25 fulltext search (trọng số 0.4)
2. Vector similarity search (trọng số 0.4)
3. Graph traversal — entity neighborhood + path search (trọng số 0.2)
4. Community-based retrieval — Louvain summaries (trọng số 0.15)
5. **WRRF fusion** (Weighted Reciprocal Rank Fusion) kết hợp 4 nguồn
6. Cross-encoder reranker (BAAI/bge-reranker-v2-m3) xếp hạng top-k

**Cross-encoder reranking cải thiện ~15% accuracy** so với không rerank (theo literature).

---

## 8. Bộ dữ liệu ViWiki-MHR

### 8.1. Thành phần

| Loại | Số lượng | Nguồn |
|------|---------|-------|
| Single-hop | ~23,000 | ViQuAD 2.0 |
| Unanswerable | ~5,000 | ViQuAD 2.0 (SQuAD 2.0 style) |
| Multi-hop | ~7,000 | KG walks + LLM rewrite |
| Adversarial unanswerable | ~1,000 | Broken-link (BRINK-style) |
| **Tổng** | **~36,000** | |

### 8.2. Sáu loại suy luận (Reasoning Types)

| Loại | Ví dụ | Số hop |
|------|-------|--------|
| Lookup | "Thủ đô Việt Nam là gì?" | 1 |
| Bridge | "Ai sáng lập công ty có trụ sở tại X?" | 2 |
| Comparison | "Ai sinh trước, A hay B?" | 2 |
| Intersection | "Tổ chức nào vừa ở Hà Nội vừa thành lập năm 1945?" | 2 |
| Temporal | "Sự kiện nào xảy ra trước khi X thành lập?" | 2-3 |
| Fan-out | "Liệt kê tất cả tác phẩm của tác giả X" | 1-2 |

### 8.3. Adversarial Unanswerables (câu hỏi không thể trả lời)

- Lấy một đường đi hợp lệ trong KG (ví dụ: A → B → C).
- **Bẻ gãy link cuối** — thay C bằng thực thể bịa đặt.
- Câu hỏi trông hợp lý nhưng **không có đáp án trong đồ thị**.
- Buộc agent phải xác minh từng bước thay vì đoán.
- **Đây là benchmark đầu tiên cho multi-hop unanswerability tiếng Việt.**

### 8.4. Pipeline sinh dữ liệu

```
KG Walks (2-hop, 3-hop, broken-link)
    → Vietnamese Question Templates (theo entity type + hop count)
    → LLM Rewrite (tùy chọn, cho tự nhiên hơn)
    → 3-stage QC: well-formedness | grounding match | deduplication
```

---

## 9. Mô hình ngôn ngữ nhỏ (SLM)

### 9.1. Cấu hình hiện tại

| Thông số | Giá trị |
|----------|---------|
| Model | AITeamVN/Vi-Qwen2-7B-RAG |
| Quantization | 4-bit NF4 |
| VRAM sử dụng | ~5.5 GB |
| Tốc độ sinh | ~30 tokens/giây |
| Lazy loading | Có (chỉ load khi cần) |

### 9.2. Tại sao chọn Vi-Qwen2-7B-RAG?

- **PhoBERT:** Encoder-only → không sinh được Cypher hay câu trả lời tự do.
- **VinaLLaMA (13B):** Quá lớn cho 8GB VRAM ở mức quantization chấp nhận được.
- **Vi-Qwen2-7B-RAG:** Dựa trên Qwen2.5-7B, fine-tune cho RAG tiếng Việt, vừa 5.5GB ở 4-bit, instruction-following mạnh.

### 9.3. Kế hoạch fine-tuning (planned)

| Giai đoạn | Phương pháp | Mục tiêu |
|-----------|-------------|----------|
| 1 | QLoRA (NF4, double quant, LoRA rank 32) | >95% Cypher thực thi được |
| 2 | DPO alignment | JSON tool-call compliance |

**Dữ liệu training:** 10-15K cặp Text2Cypher từ ViWiki-MHR.

---

## 10. Đánh giá (Evaluation)

### 10.1. Metrics

| Nhóm | Metric | Ý nghĩa |
|------|--------|---------|
| Retrieval | Hit Rate, MRR | Tìm đúng passage không? |
| Answer | Token F1, Exact Match | Câu trả lời đúng không? |
| Robustness | Abstain Accuracy | Biết từ chối khi không có đáp án? |
| Efficiency | Avg Latency | Nhanh không? |
| Groundedness | Citation Faithfulness | Trích dẫn có trung thực? |

### 10.2. Baselines so sánh

1. **Vector-only:** Dense retrieval thuần (không graph).
2. **BM25-only:** Sparse retrieval thuần.
3. **Graph-only:** Chỉ dùng Cypher traversal.
4. **Hybrid (ours):** Kết hợp cả 3 + reranking.

### 10.3. Tiêu chí thành công

| Metric | Mục tiêu |
|--------|----------|
| Token F1 (ViQuAD 2.0) | > 0.40 |
| Context Hit Rate | > 0.60 |
| Abstain Accuracy | > 0.60 |
| End-to-end Latency | < 3 giây |

---

## 11. Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|-----------|-----------|---------|
| Backend API | FastAPI | REST API, async, rate limiting |
| Knowledge Graph | Neo4j | Lưu trữ đồ thị, Cypher query |
| SLM | Vi-Qwen2-7B-RAG + bitsandbytes | Sinh Cypher, điều phối agent |
| NER | 6 backend: simple, underthesea, phonlp, phobert, videberta, wikilink | Trích xuất thực thể tiếng Việt |
| Embedding | GreenNode-Embedding-Large-VN / Gemini API | Vector hóa chunk cho dense retrieval |
| Reranker | BAAI/bge-reranker-v2-m3 | Xếp hạng lại kết quả |
| Dataset | HuggingFace Datasets | Nguồn Wikipedia dump + ViQuAD 2.0 |
| Package manager | uv | Quản lý dependency Python |

---

## 12. So sánh với công trình liên quan

| Hệ thống | Hạn chế | Chúng tôi khắc phục |
|----------|---------|---------------------|
| Microsoft GraphRAG | Enterprise-scale, không local | Chạy trên consumer hardware |
| StepChain | English-only, không NER tiếng Việt | 6 NER backend tiếng Việt |
| HisGraphRAG | Chỉ cho sách giáo khoa lịch sử | Toàn bộ Wikipedia tiếng Việt |
| CyVerACT | Không có agent loop, không tiếng Việt | ReAct agent + CyVer validation |
| GFM-RAG | Cần pre-train graph foundation model | Dùng SLM có sẵn, chỉ fine-tune adapter |

---

## 13. Đóng góp chính (Contributions)

1. **Hệ thống GraphRAG hoàn chỉnh:** Typed KG + 6 NER backend + hybrid retrieval + WRRF fusion + community detection + cross-encoder reranking + ReAct agent + citation verifier — tất cả chạy cục bộ.

2. **Bộ dữ liệu ViWiki-MHR (~36K):** 6 loại suy luận, adversarial unanswerables kiểu BRINK, 3-stage QC — benchmark đầu tiên cho multi-hop QA tiếng Việt trên KG.

3. **Pipeline SLM cục bộ:** Vi-Qwen2-7B-RAG ở 4-bit + Text2Cypher + CyVer validation — chạy trên 8GB VRAM, không cần internet.

---

## 14. Hướng phát triển

| Hạng mục | Chi tiết | Ưu tiên |
|----------|---------|---------|
| QLoRA fine-tuning | Text2Cypher chuyên biệt | Cao |
| DPO alignment | Tool-call JSON compliance | Cao |
| Scale KG | 5000+ pages | Trung bình |
| Ablation studies | So sánh từng thành phần retrieval | Trung bình |

---

## 15. Phần cứng yêu cầu

| Thành phần | Tối thiểu | Khuyến nghị |
|-----------|-----------|-------------|
| RAM | 16 GB | 32 GB |
| GPU VRAM | 8 GB (RTX 3060) | 12 GB (RTX 4060) |
| Disk | 50 GB (model + KG) | 100 GB |
| OS | Linux (Ubuntu 22.04+) | — |

---

## Thuật ngữ giải thích

| Thuật ngữ | Giải thích |
|-----------|------------|
| GraphRAG | Retrieval-Augmented Generation kết hợp đồ thị tri thức |
| KGQA | Knowledge Graph Question Answering — hỏi-đáp trên đồ thị tri thức |
| Multi-hop | Câu hỏi cần nhiều bước suy luận (qua nhiều node trong graph) |
| ReAct | Reason + Act — mô hình agent suy nghĩ rồi hành động |
| Text2Cypher | Chuyển câu hỏi ngôn ngữ tự nhiên thành truy vấn Cypher |
| CyVer | Cypher Verification — xác minh truy vấn bằng rule xác định |
| SLM | Small Language Model — mô hình ngôn ngữ nhỏ (< 10B params) |
| NF4 | Normal Float 4-bit — phương pháp lượng tử hóa 4-bit |
| QLoRA | Quantized Low-Rank Adaptation — fine-tune trên model đã quantize |
| DPO | Direct Preference Optimization — alignment không cần reward model |
| BM25 | Best Matching 25 — thuật toán tìm kiếm sparse cổ điển |
| Cross-encoder | Model đánh giá relevance bằng cách encode cả query+doc cùng lúc |
| Abstain | Từ chối trả lời khi không đủ bằng chứng |
| BRINK | Broken-link — kỹ thuật tạo câu hỏi adversarial bằng cách bẻ gãy link |
