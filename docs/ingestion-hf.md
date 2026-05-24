# Ingestion Mode: Hugging Face Wikipedia Dataset

Dataset: `wikimedia/wikipedia`

Common config format: `<dump>.<lang>` (for example `20231101.vi`, `20231101.en`, `20231101.simple`).

Default config targets Vietnamese Wikipedia (`20231101.vi`).

## Synchronous endpoint

`POST /ingest/hf`

```bash
curl -X POST "http://localhost:8000/ingest/hf" \
  -H "Content-Type: application/json" \
  -d '{"config_name":"20231101.vi","split":"train","sample_size":2,"streaming":true}'
```

## Streaming mode

Use `"streaming": true` for large language subsets to avoid loading full dataset into memory.

## Recommended for large imports

Use background jobs (`/ingest/hf/jobs`) so API remains responsive during ingestion.

See [Background Jobs](api/background-jobs.md) for async job management.

## Pipeline

Same as Wikipedia API ingestion:

1. Extract text from HF dataset records
2. Chunk text
3. NER extraction (configurable backend)
4. Embedding generation
5. Write to Neo4j with typed entities and relationships
