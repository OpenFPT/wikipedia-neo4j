# ViWiki-MHR Setup Guide

Hướng dẫn cài đặt và chạy pipeline ViWiki-MHR (Vietnamese Wikipedia Multi-Hop Reasoning).

## Tổng quan kiến trúc

```
[Python App] ──→ [Neo4j]   (Knowledge Graph database)
     │
     └──────→ [Qdrant]  (Vector database)
```

- **Docker Compose**: chạy Neo4j + Qdrant (database services)
- **Python (UV hoặc pip)**: cài thư viện cho code ứng dụng

## Yêu cầu

- Python 3.12+
- Docker Desktop (cho Neo4j + Qdrant)
- Git

---

## Bước 1: Cài thư viện Python (dùng UV)

### Cài UV

**Windows:**
Mở terminal, chạy:
```bash
pip install uv
```

**Linux/Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Cài thư viện

```bash
uv venv
uv pip install -r requirements.txt
```

Sau khi chạy xong, thư mục `.venv/` sẽ chứa toàn bộ thư viện cần thiết.

Khi chạy script, dùng `uv run` để tự động activate `.venv`:

```bash
uv run python -m scripts.download_viwiki --max-articles 100
```

---

## Bước 2: Khởi động Docker services

```bash
# Khởi động Neo4j + Qdrant
docker compose up -d

# Kiểm tra services đã sẵn sàng
docker compose ps
```

Sau khi chạy:
- Neo4j: `bolt://localhost:7687` (user: neo4j, pass: please-change-me)
- Neo4j Browser: http://localhost:7474
- Qdrant: http://localhost:6333

---

## Bước 3: Setup Neo4j schema

```bash
uv run python -m scripts.setup_neo4j_schema
```

Tạo constraints + indexes cho các node types: Article, Paragraph, Person, Organization, Location, Work, Event.

---

## Bước 4: Download dữ liệu Wikipedia

```bash
# Tải 100 bài viết (test nhanh, ~5MB)
uv run python -m scripts.download_viwiki --max-articles 100

# Tải 50,000 bài viết (đầy đủ hơn, ~1GB)
uv run python -m scripts.download_viwiki --max-articles 50000
```

Output: `data/viwiki_paragraphs.parquet`

---

## Bước 5: Chạy ingestion pipeline

```bash
uv run python -m scripts.run_ingestion --max-articles 100
```

Pipeline: Parquet → NER (trích xuất thực thể) → Entity Resolution → Ghi vào Neo4j.

---

## Bước 6: Chạy ứng dụng

### API Server (FastAPI)

```bash
uv run uvicorn src.main:app --reload --port 8000
```

### Gradio Demo

```bash
uv run python -m src.app_gradio
```

Truy cập: http://localhost:7860

---

## Makefile shortcuts

Nếu dùng `make`:

```bash
make install    # uv sync
make up         # docker compose up -d
make down       # docker compose down
make schema     # setup Neo4j schema
make download   # download wiki dump
make ingest     # run ingestion
make run        # start API server
make demo       # start Gradio demo
make check      # lint + typecheck + test
```

---

## Cấu hình

Tạo file `.env` tại root project (tùy chọn):

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=please-change-me
QDRANT_URL=http://localhost:6333
NER_BACKEND=simple
```

`NER_BACKEND` options:
- `simple`: Regex-based NER (mặc định, không cần cài thêm)
- `underthesea`: Dùng thư viện underthesea (cần `pip install underthesea`)

---

## Dừng services

```bash
docker compose down
```

Thêm `-v` để xóa data volumes:

```bash
docker compose down -v
```

---

## Deploy lên AuraDB (Cloud)

Thay vì chạy Neo4j local bằng Docker, có thể dùng Neo4j AuraDB Free tier (managed cloud).

### Tạo AuraDB instance

1. Đăng nhập https://console.neo4j.io
2. Chọn **Create Free Instance**
3. Chọn region gần nhất (Singapore/Asia)
4. Lưu lại Connection URI, Username, Password (chỉ hiện 1 lần!)

### Cấu hình `.env`

```env
NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password-from-auradb>
```

Lưu ý: AuraDB dùng scheme `neo4j+s://` (TLS encrypted), không phải `bolt://`.

### Deploy

```bash
make deploy-cloud
```

Hoặc chạy thủ công:

```bash
uv run python -m scripts.setup_neo4j_schema
uv run python -m scripts.run_ingestion
```

### Giới hạn AuraDB Free tier

- 200K nodes, 400K relationships
- Không hỗ trợ APOC (project không cần)
- Vector indexes: hỗ trợ (1024-dim cosine)
- Fulltext indexes: hỗ trợ
