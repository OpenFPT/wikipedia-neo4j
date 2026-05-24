# Ingestion Mode: Wikipedia API

Endpoint: `POST /ingest`

This mode fetches live pages from Wikipedia via the Python `wikipedia` package.

Pipeline:

1. Fetch page + summary
2. Chunk text
3. Extract entities via pluggable NER backend (`NER_BACKEND` env)
   - `simple`: regex heuristic + keyword type classification
   - `underthesea`: Vietnamese NER with BIO tagging
   - `phonlp`: PhoNLP + VnCoreNLP word segmentation
4. Generate embeddings (`EMBEDDING_BACKEND`: `gemini` or `local`)
5. Upsert Page/Chunk/Entity in Neo4j with typed relationships
6. Add `LINKS_TO` edges from page hyperlinks

Example:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"topics":["Graph database","Neo4j"]}'
```

## Entity types written

Entities are classified and stored with typed labels and relationships:

| NER Type | Node Label | Relationship |
|----------|------------|--------------|
| Person | `:Person` | `MENTIONS_PERSON` |
| Organization | `:Organization` | `MENTIONS_ORG` |
| Location | `:Location` | `MENTIONS_LOCATION` |
| Work | `:Work` | `MENTIONS_WORK` |
| Unknown | `:Entity` | `MENTIONS` |
