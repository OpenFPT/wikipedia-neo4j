# Architecture

## Data model

- `(:Page {id, title, url, summary})`
- `(:Chunk {id, text, sequence_number, embedding})`
- `(:Entity {id, name, type})`
- `(:Person {id, name, type})` (typed entity label)
- `(:Organization {id, name, type})` (typed entity label)
- `(:Location {id, name, type})` (typed entity label)
- `(:Work {id, name, type})` (typed entity label)

Entity extraction:

- `NER_BACKEND=simple` uses a regex heuristic.
- `NER_BACKEND=underthesea` uses Underthesea NER (offline) and maps results into entities.
- `NER_BACKEND=phonlp` uses PhoNLP + VnCoreNLP segmentation (offline) and maps results into entities.

Relationships:

- `(:Page)-[:HAS_CHUNK]->(:Chunk)`
- `(:Chunk)-[:MENTIONS]->(:Entity)`
- `(:Chunk)-[:MENTIONS_PERSON]->(:Person)`
- `(:Chunk)-[:MENTIONS_ORG]->(:Organization)`
- `(:Chunk)-[:MENTIONS_LOCATION]->(:Location)`
- `(:Chunk)-[:MENTIONS_WORK]->(:Work)`
- `(:Page)-[:LINKS_TO]->(:Page)`

## Components

- `src/main.py`: FastAPI app, auth/rate-limit guard, job APIs, health/readiness/metrics
- `src/ingest.py`: Wikipedia and HF ingestion pipelines
- `src/retrieve.py`: retrieval and answer assembly
- `src/llm.py`: Gemini embedding and Cypher generation/validation
- `src/neo4j_client.py`: Neo4j driver + schema/index setup + connectivity check
- `src/job_store.py`: persistent JSON store for async HF jobs

## Query behavior

1. Attempt Gemini-generated read-only Cypher.
2. Validate safety and output aliases.
3. Execute against Neo4j.
4. On invalid generation/runtime shape failure, fallback to hybrid fulltext retrieval.

## Lifecycle

The app uses FastAPI lifespan for startup/shutdown tasks.
Shutdown closes shared Neo4j driver cleanly.
