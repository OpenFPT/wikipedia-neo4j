# 5. External Interface Specifications

## 5.1 External System Interfaces

| System | Protocol | Host:Port | Purpose | Auth |
|--------|----------|-----------|---------|------|
| Neo4j 5.x | Bolt | `localhost:7687` | Knowledge graph storage and query | username/password (env: `NEO4J_USER`, `NEO4J_PASSWORD`) |
| Google Gemini API | HTTPS | `generativelanguage.googleapis.com` | Chunk embedding (768-dim), Cypher generation | API key rotation (file: `.gemini_key.txt`) |
| HuggingFace Hub | HTTPS | `huggingface.co` | Dataset download (`wikimedia/wikipedia`), model weights | HF token (optional, `HF_TOKEN` env) |
| Local GPU (CUDA 12.x) | N/A | localhost | Vi-Qwen2-7B-RAG inference, GreenNode embedding | N/A (local process) |

## 5.2 REST API Interface

Full documentation: [`docs/api/endpoints.md`](../api/endpoints.md)

### Core Endpoints

#### `POST /query`
Main question-answering endpoint.

**Request:**
```json
{
  "question": "Ai là người sáng lập trường đại học mà Ngô Bảo Châu từng học?",
  "top_k": 10,
  "mode": "agent"
}
```

**Response:**
```json
{
  "answer": "Trường Đại học Khoa học Tự nhiên được sáng lập bởi...",
  "supporting_facts": [
    {"title": "Ngô Bảo Châu", "sent_id": 3, "page_id": 12345},
    {"title": "Đại học Khoa học Tự nhiên", "sent_id": 1, "page_id": 67890}
  ],
  "context": [],
  "latency_ms": 3241
}
```

#### `POST /ingest/wikipedia`
Ingest a single Wikipedia article by title.

**Request:** `{"title": "Hồ Chí Minh"}`
**Response:** `{"status": "ok", "chunks": 12, "entities": 8}`

#### `POST /ingest/huggingface`
Start async bulk ingestion from HuggingFace dataset.

**Request:** `{"dataset": "wikimedia/wikipedia", "limit": 1000}`
**Response:** `{"job_id": "hf-abc123", "status": "running"}`

#### `GET /ingest/jobs/{job_id}`
Check ingestion job status.

**Response:** `{"job_id": "hf-abc123", "status": "running", "processed": 450, "total": 1000}`

#### `GET /health`
Liveness check. Returns 200 if server is running.

#### `GET /ready`
Readiness check. Returns 200 only if Neo4j connection is healthy.

#### `GET /metrics`
Prometheus-compatible metrics endpoint.

### Authentication

Protected endpoints require `X-API-Key` header when `APP_API_KEY` is set in `.env`:
```
X-API-Key: your-api-key-here
```
Returns `401 Unauthorized` if key is missing or invalid.

## 5.3 Data File Interfaces

### Input Formats

| Format | Used By | Schema |
|--------|---------|--------|
| Arrow/Parquet | HuggingFace dataset reader | `{id, url, title, text}` |
| JSONL | Bulk chunk export | `{chunk_id, page_title, page_url, text, sequence_number}` |
| CSV | Link export | `{source_title, target_title}` |
| JSONL | Entity export | `{name, type, page_title}` |

### Output Formats

| Format | Produced By | Schema |
|--------|------------|--------|
| JSONL | Evaluation pipeline | `{question, answer, retrieved_chunks[], latency_ms, hit, rr}` |
| JSONL | MHQA dataset generation | See `template.json` for full schema |
| JSON | API responses | Per-endpoint schema above |

## 5.4 Configuration Interface

All configuration via environment variables in `.env` file. Key variables:

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<password>

# Embedding
EMBEDDING_BACKEND=local          # gemini | local
EMBEDDING_DIM=1024

# NER
NER_BACKEND=wikilink             # simple | underthesea | wikilink | videberta

# LLM
MODEL_MODE=local                 # api | local
LOCAL_MODEL_ID=AITeamVN/Vi-Qwen2-7B-RAG

# WRRF weights
WRRF_WEIGHT_BM25=0.4
WRRF_WEIGHT_VECTOR=0.4
WRRF_WEIGHT_GRAPH=0.2
WRRF_WEIGHT_COMMUNITY=0.15

# Security
APP_API_KEY=                     # empty = no auth
RATE_LIMIT_PER_MINUTE=120
```
