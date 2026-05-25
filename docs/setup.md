# Setup & Run

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Neo4j (installed as systemd service)
- (Optional) NVIDIA GPU with CUDA for local model mode

## 1) Configure environment

```bash
cp .env.example .env
printf "%s" "YOUR_GEMINI_API_KEY" > .gemini_key.txt
chmod 600 .gemini_key.txt
```

Important envs:

- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- `NER_BACKEND`: `simple` (default), `underthesea`, or `phonlp`
- `EMBEDDING_BACKEND`: `gemini` (default) or `local`
- `MODEL_MODE`: `api` (default, uses Gemini) or `local` (uses Qwen2.5-7B)
- `LOCAL_MODEL_ID`: model identifier (default `Qwen/Qwen2.5-7B-Instruct`)
- Optional: `APP_API_KEY`, `RATE_LIMIT_PER_MINUTE`, `LOG_LEVEL`
- Optional strict startup check: `REQUIRE_GEMINI_KEY_ON_STARTUP=true`

## 2) Start Neo4j

Neo4j runs as a systemd service on this machine:

```bash
sudo systemctl start neo4j
sudo systemctl status neo4j   # verify it's running
```

Default connection: `bolt://localhost:7687` with auth configured in `/etc/neo4j/neo4j.conf`.

## 3) Install dependencies

```bash
uv sync --all-groups
```

For PhoNLP backend, models are auto-downloaded on first use to `.phonlp/` and `.vncorenlp/`.

## 4) Run API

```bash
uv run uvicorn src.main:app --reload --port 8000
```

Open:

- API docs: <http://localhost:8000/docs>
- Neo4j Browser: <http://localhost:7474>

## 5) Local quality checks

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
python -m compileall -q src tests
```

## 6) Dataset generation (optional)

For reproducible Vietnamese Wikipedia inputs, use the processed HF snapshot
`Keithsel/viwiki-20260523`. It is generated from raw MediaWiki XML with
`scripts/viwiki_processing/`, which exports cleaned and raw Parquet shards.

```bash
uv sync --group xml-processing
python -m scripts.viwiki_processing.cli convert \
  --xml dumps/viwiki-20260523-pages-articles.xml \
  --output articles_cleaned \
  --raw-output articles_raw \
  --limit 1000  # optional smoke-test limit
```

After ingesting data, generate the ViWiki-MHR QA dataset:

```python
from src.dataset_gen import generate_dataset

stats = generate_dataset(
    two_hop_limit=5000,
    three_hop_limit=3000,
    broken_limit=1000,
    output_path="data/viwiki_mhr.jsonl",
    rewrite=True,   # use local LLM to rewrite questions
    qc=True,        # run QC pipeline
)
```

Output is written to `data/viwiki_mhr.jsonl`.
