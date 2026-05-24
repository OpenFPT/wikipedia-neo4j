# Trình tự thực hiện từng bước

## Bước 1: Verify chunk IDs có match không

```bash
# Lấy 1 chunk ID từ dataset
head -1 data/viwiki_mhr.jsonl | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['metadata']['evidence_chunk_ids'][0])"

# Check trong Neo4j
cypher-shell -u neo4j -p "M_sE>>'S3^f#d:P" \
  "MATCH (c:Chunk {id: 'd26c2be4-4251-576a-aac5-63c63fc1afaf'}) RETURN c.id, left(c.text, 80)"
```

---

## Bước 2A: Nếu chunk IDs KHÔNG match (khả năng cao)

Tạo test set mới từ chunks thực tế trong Neo4j:

```bash
# Xem format chunk ID trong Neo4j
cypher-shell -u neo4j -p "M_sE>>'S3^f#d:P" \
  "MATCH (c:Chunk) RETURN c.id LIMIT 5"
```

Sau đó viết script tạo test set mới — lấy câu hỏi từ `viwiki_mhr.jsonl` + map lại gold chunk IDs dựa trên `source_pages` trong metadata.

## Bước 2B: Nếu chunk IDs match

Chạy evaluation lớn hơn:
```bash
uv run python -m src.evaluation 500
```

---

## Bước 3: Phân tích kết quả baseline

Xem file `reports/eval_results.json` — kiểm tra:
- Bao nhiêu câu có hit (tìm được chunk đúng)?
- Reranking có cải thiện không?
- Câu nào fail? Tại sao?

---

## Bước 4: Implement query decomposition

Thêm module tách câu hỏi multi-hop thành sub-questions:
- Input: "Ai sáng lập tổ chức mà X là thành viên?"
- Output: ["X là thành viên tổ chức nào?", "Ai sáng lập tổ chức Y?"]
- Mỗi sub-question query riêng → merge kết quả

---

## Bước 5: Community detection

```bash
# Cài Neo4j GDS plugin (nếu chưa có)
# Chạy Louvain trên graph
cypher-shell -u neo4j -p "M_sE>>'S3^f#d:P" \
  "CALL gds.louvain.stream('myGraph') YIELD nodeId, communityId RETURN communityId, count(*) ORDER BY count(*) DESC LIMIT 20"
```

Tạo summary cho mỗi community bằng LLM.

---

## Bước 6: Evaluation round 2

Chạy lại evaluation sau khi thêm query decomposition + community detection. So sánh với baseline.

---

## Bước 7: Neo4j vector index

```cypher
CREATE VECTOR INDEX chunk_embeddings FOR (c:Chunk) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
```

---

## Việc cần làm NGAY: Chạy bước 1, gửi kết quả để xác định đi bước 2A hay 2B.
