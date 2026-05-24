# Operations & Troubleshooting

## Common issues

### Neo4j auth mismatch

- Symptom: `Neo.ClientError.Security.Unauthorized`
- Fix: ensure `.env` `NEO4J_PASSWORD` matches the password configured in `/etc/neo4j/neo4j.conf`.

### Empty retrieval after ingestion

- Ensure ingestion actually completed.
- Retry with direct page-title keywords.

### Gemini region/quota errors

- Multi-key rotation is supported via `.gemini_key.txt`.
- If all keys fail, ingestion/query generation may fail and query falls back.

### Background jobs after restart

- Jobs are persisted in `.hf_ingest_jobs.json`.
- Jobs active during restart are marked `interrupted`.

### Local model OOM

- Symptom: CUDA out of memory when `MODEL_MODE=local`.
- Fix: ensure GPU has at least 6GB VRAM for 4-bit quantized Qwen2.5-7B.
- Alternative: reduce `max_new_tokens` or use API mode.

### Agent not converging

- The ReAct agent has a 6-iteration limit.
- If answers are poor, check that fulltext indexes exist (`chunk_text_ft`, `page_title_ft`).
- Verify graph has sufficient data ingested.

## Security controls

Optional:

- `APP_API_KEY` to require `X-API-Key` on protected endpoints.
- `RATE_LIMIT_PER_MINUTE` for per-client request throttling.

## Logging

- `LOG_LEVEL` controls verbosity (`DEBUG`, `INFO`, `WARNING`, ...).
- `JSON_LOGS=true` enables structured one-line JSON logs.
- Log records include `request_id` and relevant operational fields when available (e.g. `duration_ms`, `job_id`, `status`).

## Error payloads

All API errors return a consistent schema:

```json
{
  "error_code": "invalid_request",
  "message": "Unauthorized",
  "request_id": "c2c57f7b-1b2b-4a7c-8b4f-9ae6e1f1f5c2",
  "hint": "Provide a valid X-API-Key header"
}
```

Stable error codes:

- `cypher_generation_failed`
- `ingest_failed`
- `invalid_request`
- `key_config_invalid`

Use `request_id` to correlate error responses with logs.

## Readiness and metrics

- `GET /ready` verifies Neo4j connectivity and reports dependency status.
- `GET /metrics` exposes `hf_jobs_total{status=...}`.

## Useful commands

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
python -m compileall -q src tests
uv run mkdocs build
```

## About MkDocs Material warning banner

Material may print an informational warning about MkDocs 2.0 changes.

- Not a build failure.
- Project pins compatible versions (`mkdocs<2`, `mkdocs-material<10`).

Optional quiet build:

```bash
NO_MKDOCS_2_WARNING=1 uv run mkdocs build
```
