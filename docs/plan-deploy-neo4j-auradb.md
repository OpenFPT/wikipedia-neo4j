# Plan: Deploy Neo4j lên AuraDB Free Tier

## Context
Nhánh feat/pipeline-and-tools đã merge vào main. Task tiếp theo là deploy Neo4j lên cloud để team có thể truy cập shared database thay vì chạy local Docker. AuraDB Free tier được chọn vì miễn phí, managed, và project đã tương thích sẵn (không dùng APOC).

## AuraDB Free Tier Limits
- 200K nodes, 400K relationships
- 1 database
- Vector indexes: supported (1024-dim OK)
- Fulltext indexes: supported
- APOC: không có — nhưng project không cần

## Các bước thực hiện

### Bước 1: Tạo AuraDB Free Instance
- Đăng nhập https://console.neo4j.io
- Tạo Free instance, chọn region gần (Singapore hoặc Asia)
- Lưu lại: Connection URI, Username, Password (chỉ hiện 1 lần)

### Bước 2: Cập nhật `.env` cho AuraDB
Thay đổi 3 biến:
```env
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password-from-auradb>
```

### Bước 3: Setup schema trên AuraDB
```bash
uv run python -m scripts.setup_neo4j_schema
```
Hoặc dùng `neo4j_client.py` setup_schema() — tạo constraints, indexes, vector indexes.

File liên quan: `src/neo4j_client.py` (lines 49-120), `scripts/setup_neo4j_schema.py`

### Bước 4: Ingest data lên AuraDB
```bash
uv run python -m scripts.run_ingestion
```
File liên quan: `scripts/run_ingestion.py`, `src/ingest.py`

### Bước 5: Thêm Makefile target cho cloud deploy
Thêm target `deploy-cloud` vào `Makefile` để document quy trình:
```makefile
deploy-cloud:
	@echo "Ensure .env has NEO4J_URI=neo4j+s://... pointing to AuraDB"
	uv run python -m scripts.setup_neo4j_schema
	uv run python -m scripts.run_ingestion
```

### Bước 6: Cập nhật `.env.example` và docs
- Thêm comment hướng dẫn AuraDB URI format vào `.env.example`
- Cập nhật `SETUP_GUIDE.md` thêm section AuraDB

## Files cần sửa
| File | Thay đổi |
|------|----------|
| `.env` | Cập nhật credentials (KHÔNG commit) |
| `.env.example` | Thêm comment AuraDB format |
| `Makefile` | Thêm target deploy-cloud |
| `SETUP_GUIDE.md` | Thêm hướng dẫn AuraDB |

## Không cần sửa code
- `src/neo4j_client.py` — driver đã dùng `settings.neo4j_uri` trực tiếp, hỗ trợ `neo4j+s://`
- `src/retrieve.py` — vector search dùng `db.index.vector.queryNodes`, AuraDB hỗ trợ
- `src/config.py` — đọc env vars, không cần thay đổi

## Verification
1. Verify kết nối: `uv run python -c "from src.neo4j_client import Neo4jClient; c = Neo4jClient(); print(c.get_server_version())"`
2. Verify schema: chạy `make schema` — check indexes được tạo
3. Verify ingestion: chạy ingestion một batch nhỏ, check data trên AuraDB console
4. Verify query: `uv run python -m src.main` với một câu hỏi đơn giản
